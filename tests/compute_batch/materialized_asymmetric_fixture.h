#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "hyle/graph/batch.h"
#include "hyle/simd/dtype.h"

namespace materialized_asymmetric_test {
namespace c = hyle::compute;
namespace gb = hyle::graph::batch;

inline void exercise(c::Context &context) {
  for (const bool paged : {false, true})
    for (const bool bf16 : {false, true})
      for (const bool mla : {false, true}) {
        SCOPED_TRACE(::testing::Message() << paged << ':' << bf16 << ':' << mla);
        const auto type = bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat16;
        const unsigned d = mla ? 576 : 64, dv = mla ? 512 : 128;
        const unsigned heads = mla ? 64 : 2;
        const c::BatchCapacity cap{
            .storage = paged ? c::BatchStorage::kPaged : c::BatchStorage::kRagged,
            .query_type = type, .output_type = type,
            .key_storage = c::CacheStorage::plain(type).value(),
            .value_storage = c::CacheStorage::plain(type).value(),
            .max_requests = 2, .max_query_rows = 4, .max_kv_rows = 6,
            .max_pages = paged ? 3u : 0u, .max_page_refs = paged ? 3u : 0u,
            .query_heads = heads, .kv_heads = 1, .qk_head_dim = d,
            .value_head_dim = dv, .page_size = paged ? 2u : 1u};
        const auto schema = c::BatchSchema::create(cap).value();
        const std::array<std::int32_t, 3> qp{0, 2, 3};
        const std::array<std::int32_t, 3> kp{0, paged ? 2 : 3, paged ? 3 : 5};
        const std::array<std::int32_t, 3> pages{2, 0, 1};
        const std::array<std::int32_t, 2> tails{1, 2};
        const auto layout = schema.version({qp, kp,
            paged ? std::span(pages) : std::span<const std::int32_t>{},
            paged ? std::span(tails) : std::span<const std::int32_t>{}}).value();
        const auto recipe = c::MaterializedBatchProgram::prepare(schema, {
            .max_kv_per_request = 3, .max_host_metadata_bytes = 1u << 20,
            .max_logical_intermediate_bytes = 32u << 20,
            .scale = 1.F / std::sqrt(float(d))}).value();
        const auto output_desc = recipe.descriptions()[recipe.output_value()];
        ASSERT_NE(output_desc, recipe.descriptions()[0]);
        auto queue = context.create_queue().value();
        auto graph = gb::MaterializedInvocation::create(context, recipe, queue,
            {.max_output_bytes = *output_desc.required_bytes() + 64}).value();
        auto direct = c::MaterializedBatchInvocation::create(context, recipe, queue,
            {.max_payload_bytes = 32u << 20, .max_workspace_bytes = 32u << 20}).value();
        auto pool = c::LeasePool::create(hyle::mem::system_resource(), 8u << 20).value();
        const auto encode = [bf16](float x) {
          return bf16 ? hyle::simd::f32_to_bf16(x).bits : hyle::simd::f32_to_f16(x).bits;
        };
        const auto decode = [bf16](std::uint16_t x) {
          return bf16 ? hyle::simd::bf16_to_f32({x}) : hyle::simd::f16_to_f32({x});
        };
        const auto load = [&](std::span<const std::uint16_t> words) {
          auto writer = pool.allocate(words.size_bytes()).value();
          EXPECT_TRUE(writer.view()->write(std::as_bytes(words)).ok());
          return writer.publish().value();
        };
        for (unsigned phase = 0; phase < 2; ++phase) {
          std::vector<std::uint16_t> q(std::size_t(4) * heads * d), k(6 * d), v(6 * dv);
          for (std::size_t i = 0; i < q.size(); ++i)
            q[i] = encode(.2F * std::sin(float(i + phase * 17) * .017F));
          for (std::size_t i = 0; i < k.size(); ++i)
            k[i] = encode(.3F * std::cos(float(i + phase * 13) * .019F));
          for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = encode(std::sin(float(i + phase * 23) * .013F));
          auto query = load(q), key = load(k), value = load(v);
          auto keys = c::PagedArena{}, values = c::PagedArena{};
          std::array<c::ReadLease, 3> key_pages, value_pages;
          auto kr = c::PagedArenaRead{}, vr = c::PagedArenaRead{};
          if (paged) {
            keys = c::PagedArena::create(schema, c::BatchCacheRole::kKey,
                hyle::mem::system_resource(), {.max_backing_bytes = k.size() * 2 + 256}).value();
            values = c::PagedArena::create(schema, c::BatchCacheRole::kValue,
                hyle::mem::system_resource(), {.max_backing_bytes = v.size() * 2 + 256}).value();
            for (std::size_t i = 0; i < pages.size(); ++i) {
              auto kw = keys.acquire(pages[i]).value(), vw = values.acquire(pages[i]).value();
              ASSERT_TRUE(kw.lease().view()->write(std::as_bytes(
                  std::span(k).subspan(std::size_t(pages[i]) * 2 * d, 2 * d))).ok());
              ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(
                  std::span(v).subspan(std::size_t(pages[i]) * 2 * dv, 2 * dv))).ok());
              key_pages[i] = kw.publish(i == 1 ? 1 : 2).value();
              value_pages[i] = vw.publish(i == 1 ? 1 : 2).value();
            }
            kr = keys.read(layout, key_pages).value();
            vr = values.read(layout, value_pages).value();
          }
          ASSERT_TRUE(graph.prepare_batch(layout).ok());
          ASSERT_TRUE(direct.prepare_batch(layout).ok());
          ASSERT_TRUE((paged ? graph.bind(query, kr, vr) : graph.bind(query, key, value)).ok());
          ASSERT_TRUE((paged ? direct.bind(query, kr, vr) : direct.bind(query, key, value)).ok());
          auto gdone = graph.submit().value(), ddone = direct.submit().value();
          ASSERT_TRUE(gdone.wait().ok());
          ASSERT_TRUE(ddone.wait().ok());
          auto gout = graph.output().value();
          const std::array outputs{gout.storage_lease().value(), direct.output().value()};
          for (const auto &output : outputs) {
            std::vector<std::uint16_t> actual(*output_desc.element_count());
            ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
            for (unsigned request = 0; request < 2; ++request) {
              const unsigned length = request == 0 ? 3 : 2;
              const auto physical = [&](unsigned row) {
                return paged ? unsigned(pages[kp[request] + row / 2]) * 2 + row % 2
                             : unsigned(kp[request]) + row;
              };
              for (int token = qp[request]; token < qp[request + 1]; ++token)
                for (unsigned head = 0; head < heads; ++head) {
                  const auto visible = length - unsigned(qp[request + 1] - token) + 1;
                  std::vector<double> scores(visible);
                  for (unsigned row = 0; row < visible; ++row)
                    for (unsigned col = 0; col < d; ++col)
                      scores[row] += double(decode(q[(std::size_t(token) * heads + head) * d + col])) *
                          decode(k[physical(row) * d + col]) / std::sqrt(double(d));
                  const auto maximum = *std::max_element(scores.begin(), scores.end());
                  double total = 0;
                  for (auto &score : scores) total += score = std::exp(score - maximum);
                  for (unsigned col = 0; col < dv; ++col) {
                    double expected = 0;
                    for (unsigned row = 0; row < visible; ++row)
                      expected += scores[row] / total * decode(v[physical(row) * dv + col]);
                    EXPECT_NEAR(decode(actual[(std::size_t(token) * heads + head) * dv + col]),
                                expected, bf16 ? .007 : .002);
                  }
                }
            }
          }
          EXPECT_FALSE(graph.prepare_batch(layout).ok());
          EXPECT_FALSE(direct.prepare_batch(layout).ok());
        }
      }
}
} // namespace materialized_asymmetric_test
