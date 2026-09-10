#include "hyle/io/tls.h"

#if defined(HYLE_IO_TLS)

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>
#include <openssl/x509err.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "hyle/sched/cancel.h"

namespace hyle::io {
namespace detail {

struct TlsCredentialAccess {
  static constexpr container::StringView
  borrowed_value(const TlsPrivateKeyPassword &password) noexcept {
    return password.borrowed_value();
  }

  static const TlsPasswordCallback *
  callback_ref(const TlsPrivateKeyPassword &password) noexcept {
    return password.callback_ref();
  }
};

} // namespace detail
namespace {

constexpr std::size_t kPumpBytes = 16 * 1024;
// Opportunistic input exists only to break a full-duplex transport stalemate
// while one TLS record is being flushed. Two pump chunks cover one maximum TLS
// record plus provider overhead without turning a write operation into an
// unbounded peer-input queue.
constexpr std::size_t kMaxRetainedCiphertextBytes = 2 * kPumpBytes;

struct ContextImpl {
  std::atomic<std::uint32_t> references{1};
  SSL_CTX *provider = nullptr;
  unsigned char *alpn = nullptr;
  unsigned int alpn_size = 0;
  std::uint64_t identity = 0;
  bool server = false;
  bool verify_peer = false;
  bool require_alpn = false;
  bool allow_truncated_eof = false;
  mem::AllocatorRef connection_allocator{};
};

std::atomic<std::uint64_t> next_context_identity{1};

ContextImpl *as_context(void *value) noexcept {
  return static_cast<ContextImpl *>(value);
}

SSL *as_ssl(void *value) noexcept { return static_cast<SSL *>(value); }
BIO *as_bio(void *value) noexcept { return static_cast<BIO *>(value); }

void retain_context(ContextImpl *context) noexcept {
  context->references.fetch_add(1, std::memory_order_relaxed);
}

void release_context(ContextImpl *context) noexcept {
  if (context == nullptr ||
      context->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
    return;
  }
  SSL_CTX_free(context->provider);
  delete[] context->alpn;
  delete context;
}

class OwnedCString {
public:
  OwnedCString() noexcept = default;
  ~OwnedCString() { delete[] owned_; }
  OwnedCString(const OwnedCString &) = delete;
  OwnedCString &operator=(const OwnedCString &) = delete;

  Status assign(container::StringView value, bool required,
                const char *field) noexcept {
    if (value.empty()) {
      if (required)
        return Status::InvalidArgument(field);
      return {};
    }
    if (value.data() == nullptr ||
        value.size() >
            static_cast<std::uint32_t>((std::numeric_limits<int>::max)())) {
      return Status::InvalidArgument(field);
    }
    if (std::memchr(value.data(), '\0', value.size()) != nullptr) {
      return Status::InvalidArgument("TLS path/name contains an embedded NUL");
    }
    if (value.is_null_terminated()) {
      data_ = value.data();
      return {};
    }
    owned_ =
        new (std::nothrow) char[static_cast<std::size_t>(value.size()) + 1];
    if (owned_ == nullptr) {
      return Status::ResourceExhausted("TLS option copy allocation failed");
    }
    std::memcpy(owned_, value.data(), value.size());
    owned_[value.size()] = '\0';
    data_ = owned_;
    return {};
  }

  const char *get() const noexcept { return data_; }

private:
  const char *data_ = nullptr;
  char *owned_ = nullptr;
};

template <class T, auto Release> struct ProviderDeleter {
  void operator()(T *value) const noexcept {
    if (value != nullptr)
      (void)Release(value);
  }
};

using BioOwner = std::unique_ptr<BIO, ProviderDeleter<BIO, BIO_free>>;
using CertificateOwner =
    std::unique_ptr<X509, ProviderDeleter<X509, X509_free>>;
using PrivateKeyOwner =
    std::unique_ptr<EVP_PKEY, ProviderDeleter<EVP_PKEY, EVP_PKEY_free>>;

constexpr std::size_t kMaxIdentityCertificateCount = 64;

static_assert(std::is_trivially_copyable_v<TlsCredentialSource>);
static_assert(std::is_trivially_copyable_v<TlsPrivateKeyPassword>);

void clear_provider_errors() noexcept {
  while (ERR_get_error() != 0) {
  }
}

Status invalid_credential(const char *message) noexcept {
  bool allocation_failed = false;
  for (unsigned long error = ERR_get_error(); error != 0;
       error = ERR_get_error()) {
    allocation_failed |= ERR_GET_REASON(error) == ERR_R_MALLOC_FAILURE;
  }
  return allocation_failed ? Status::ResourceExhausted(
                                 "TLS provider credential allocation failed")
                           : Status::InvalidArgument(message);
}

Status validate_credential_source(const TlsCredentialSource &source,
                                  const char *missing_message) noexcept {
  if (!source.supplied())
    return Status::InvalidArgument(missing_message);
  if (source.kind() != TlsCredentialSourceKind::kFile &&
      source.kind() != TlsCredentialSourceKind::kPem) {
    return Status::InvalidArgument("invalid TLS credential source kind");
  }
  const container::StringView value = source.value();
  if (value.empty() || value.data() == nullptr) {
    return Status::InvalidArgument(missing_message);
  }
  if (value.size() >
      static_cast<std::uint32_t>((std::numeric_limits<int>::max)())) {
    return Status::OutOfRange("TLS credential source is too large");
  }
  if (std::memchr(value.data(), '\0', value.size()) != nullptr) {
    return Status::InvalidArgument(
        source.kind() == TlsCredentialSourceKind::kFile
            ? "TLS credential path contains an embedded NUL"
            : "TLS PEM contains an embedded NUL");
  }
  return {};
}

Status open_credential_bio(const TlsCredentialSource &source,
                           const char *missing_message,
                           BioOwner &result) noexcept {
  Status status = validate_credential_source(source, missing_message);
  if (!status.ok())
    return status;

  ERR_clear_error();
  if (source.kind() == TlsCredentialSourceKind::kFile) {
    OwnedCString path;
    status = path.assign(source.value(), true, missing_message);
    if (!status.ok())
      return status;
    result.reset(BIO_new_file(path.get(), "rb"));
    if (!result) {
      return invalid_credential("TLS credential file could not be opened");
    }
  } else {
    result.reset(BIO_new_mem_buf(source.value().data(),
                                 static_cast<int>(source.value().size())));
    if (!result) {
      clear_provider_errors();
      return Status::ResourceExhausted(
          "TLS credential memory BIO allocation failed");
    }
  }
  return {};
}

int reject_pem_password(char *, int, int, void *) noexcept { return 0; }

bool reached_pem_end() noexcept {
  const unsigned long error = ERR_peek_last_error();
  const bool finished =
      error == 0 || (ERR_GET_LIB(error) == ERR_LIB_PEM &&
                     ERR_GET_REASON(error) == PEM_R_NO_START_LINE);
  clear_provider_errors();
  return finished;
}

Status load_certificate_chain(SSL_CTX *provider,
                              const TlsCredentialSource &source) noexcept {
  BioOwner bio;
  Status status = open_credential_bio(
      source, "TLS certificate chain source is required", bio);
  if (!status.ok())
    return status;

  ERR_clear_error();
  CertificateOwner leaf(
      PEM_read_bio_X509_AUX(bio.get(), nullptr, &reject_pem_password, nullptr));
  if (!leaf) {
    return invalid_credential("TLS certificate chain is not valid PEM");
  }
  if (SSL_CTX_use_certificate(provider, leaf.get()) != 1 ||
      SSL_CTX_clear_chain_certs(provider) != 1) {
    return invalid_credential("TLS certificate could not be installed");
  }

  std::size_t count = 1;
  while (count < kMaxIdentityCertificateCount) {
    CertificateOwner certificate(
        PEM_read_bio_X509(bio.get(), nullptr, &reject_pem_password, nullptr));
    if (!certificate) {
      return reached_pem_end()
                 ? Status{}
                 : invalid_credential("TLS certificate chain is malformed");
    }
    if (SSL_CTX_add1_chain_cert(provider, certificate.get()) != 1) {
      return invalid_credential(
          "TLS certificate chain entry could not be installed");
    }
    ++count;
  }

  CertificateOwner overflow(
      PEM_read_bio_X509(bio.get(), nullptr, &reject_pem_password, nullptr));
  if (overflow) {
    return invalid_credential(
        "TLS certificate chain exceeds the 64-certificate limit");
  }
  return reached_pem_end()
             ? Status{}
             : invalid_credential("TLS certificate chain is malformed");
}

bool duplicate_authority_error(unsigned long error) noexcept {
  return ERR_GET_LIB(error) == ERR_LIB_X509 &&
         ERR_GET_REASON(error) == X509_R_CERT_ALREADY_IN_HASH_TABLE;
}

Status
load_certificate_authorities(SSL_CTX *provider,
                             const TlsCredentialSource &source) noexcept {
  BioOwner bio;
  Status status = open_credential_bio(
      source, "TLS certificate authority source is required", bio);
  if (!status.ok())
    return status;
  X509_STORE *store = SSL_CTX_get_cert_store(provider);
  if (store == nullptr) {
    clear_provider_errors();
    return Status::ResourceExhausted(
        "TLS certificate authority store is unavailable");
  }

  bool added = false;
  ERR_clear_error();
  for (;;) {
    CertificateOwner certificate(PEM_read_bio_X509_AUX(
        bio.get(), nullptr, &reject_pem_password, nullptr));
    if (!certificate) {
      if (added && reached_pem_end())
        return {};
      return invalid_credential(
          "TLS certificate authority bundle is malformed");
    }
    if (X509_STORE_add_cert(store, certificate.get()) != 1) {
      const unsigned long error = ERR_peek_last_error();
      if (!duplicate_authority_error(error)) {
        return invalid_credential(
            "TLS certificate authority could not be installed");
      }
      clear_provider_errors();
    }
    added = true;
  }
}

Status
validate_private_key_password(const TlsPrivateKeyPassword &password) noexcept {
  switch (password.kind()) {
  case TlsPrivateKeyPasswordKind::kNone:
    return {};
  case TlsPrivateKeyPasswordKind::kBorrowed: {
    const container::StringView value =
        detail::TlsCredentialAccess::borrowed_value(password);
    if (value.data() == nullptr && !value.empty()) {
      return Status::InvalidArgument(
          "TLS private key password has no backing storage");
    }
    if (value.size() >
        static_cast<std::uint32_t>((std::numeric_limits<int>::max)())) {
      return Status::OutOfRange("TLS private key password is too large");
    }
    return {};
  }
  case TlsPrivateKeyPasswordKind::kCallback:
    return detail::TlsCredentialAccess::callback_ref(password) != nullptr
               ? Status{}
               : Status::InvalidArgument(
                     "TLS private key password callback is missing");
  }
  return Status::InvalidArgument("invalid TLS private key password kind");
}

struct PasswordAdapter {
  const TlsPrivateKeyPassword *source = nullptr;
  Status failure{};
};

int provide_private_key_password(char *buffer, int capacity, int,
                                 void *opaque) noexcept {
  auto *adapter = static_cast<PasswordAdapter *>(opaque);
  if (buffer == nullptr || capacity <= 0 || adapter == nullptr ||
      adapter->source == nullptr || !adapter->failure.ok()) {
    return 0;
  }

  container::StringView password;
  switch (adapter->source->kind()) {
  case TlsPrivateKeyPasswordKind::kNone:
    return 0;
  case TlsPrivateKeyPasswordKind::kBorrowed:
    password = detail::TlsCredentialAccess::borrowed_value(*adapter->source);
    break;
  case TlsPrivateKeyPasswordKind::kCallback: {
    const TlsPasswordCallback *callback =
        detail::TlsCredentialAccess::callback_ref(*adapter->source);
    if (callback == nullptr) {
      adapter->failure = Status::InvalidArgument(
          "TLS private key password callback is missing");
      return 0;
    }
    Result<container::StringView> supplied = (*callback)();
    if (!supplied) {
      adapter->failure = static_cast<Status>(std::move(supplied).error());
      return 0;
    }
    password = *supplied;
    break;
  }
  }

  if (password.data() == nullptr && !password.empty()) {
    adapter->failure = Status::InvalidArgument(
        "TLS private key password callback returned an invalid view");
    return 0;
  }
  if (password.size() >= static_cast<std::uint32_t>(capacity)) {
    adapter->failure = Status::OutOfRange(
        "TLS private key password exceeds the provider buffer");
    return 0;
  }
  if (!password.empty())
    std::memcpy(buffer, password.data(), password.size());
  buffer[password.size()] = '\0';
  return static_cast<int>(password.size());
}

Status load_private_key(SSL_CTX *provider, const TlsCredentialSource &source,
                        const TlsPrivateKeyPassword &password) noexcept {
  Status status = validate_private_key_password(password);
  if (!status.ok())
    return status;

  BioOwner bio;
  status =
      open_credential_bio(source, "TLS private key source is required", bio);
  if (!status.ok())
    return status;

  PasswordAdapter adapter{&password};
  ERR_clear_error();
  PrivateKeyOwner key(PEM_read_bio_PrivateKey(
      bio.get(), nullptr, &provide_private_key_password, &adapter));
  if (!adapter.failure.ok()) {
    clear_provider_errors();
    return std::move(adapter.failure);
  }
  if (!key) {
    return invalid_credential(
        "TLS private key is invalid or could not be decrypted");
  }
  if (SSL_CTX_use_PrivateKey(provider, key.get()) != 1) {
    return invalid_credential("TLS private key could not be installed");
  }
  clear_provider_errors();
  return {};
}

Status load_identity(SSL_CTX *provider,
                     const TlsCredentialSource &certificate_chain,
                     const TlsCredentialSource &private_key,
                     const TlsPrivateKeyPassword &password) noexcept {
  Status status = load_certificate_chain(provider, certificate_chain);
  if (status.ok())
    status = load_private_key(provider, private_key, password);
  if (status.ok() && SSL_CTX_check_private_key(provider) != 1) {
    status = invalid_credential("TLS certificate and private key do not match");
  }
  return status;
}

Status validate_client_credentials(const TlsClientOptions &options) noexcept {
  const bool has_certificate = options.certificate_chain.supplied();
  const bool has_key = options.private_key.supplied();
  if (has_certificate != has_key) {
    return Status::InvalidArgument(
        "TLS client certificate and private key must be supplied together");
  }
  if (!has_key && options.private_key_password.supplied()) {
    return Status::InvalidArgument(
        "TLS private key password requires a private key");
  }
  if (options.verify == TlsVerifyMode::kNone &&
      options.certificate_authority.supplied()) {
    return Status::InvalidArgument(
        "TLS certificate authority requires peer verification");
  }
  return validate_private_key_password(options.private_key_password);
}

Status validate_server_credentials(const TlsServerOptions &options) noexcept {
  if (!options.certificate_chain.supplied() ||
      !options.private_key.supplied()) {
    return Status::InvalidArgument(
        "TLS server certificate and private key are required");
  }
  if (options.client_verification == TlsVerifyMode::kPeer &&
      !options.client_certificate_authority.supplied()) {
    return Status::InvalidArgument(
        "TLS client certificate authority is required for peer verification");
  }
  if (options.client_verification == TlsVerifyMode::kNone &&
      options.client_certificate_authority.supplied()) {
    return Status::InvalidArgument(
        "TLS client certificate authority requires peer verification");
  }
  return validate_private_key_password(options.private_key_password);
}

int provider_version(TlsVersion version) noexcept {
  switch (version) {
  case TlsVersion::kProviderDefault:
    return 0;
  case TlsVersion::kTls12:
    return TLS1_2_VERSION;
  case TlsVersion::kTls13:
    return TLS1_3_VERSION;
  }
  return -1;
}

Status configure_versions(SSL_CTX *provider, TlsVersion minimum,
                          TlsVersion maximum) noexcept {
  const int min_value = provider_version(minimum);
  const int max_value = provider_version(maximum);
  if (min_value < 0 || max_value < 0 ||
      (min_value != 0 && max_value != 0 && min_value > max_value)) {
    return Status::InvalidArgument("invalid TLS protocol version range");
  }
  if (SSL_CTX_set_min_proto_version(provider, min_value) != 1 ||
      SSL_CTX_set_max_proto_version(provider, max_value) != 1) {
    return Status::InvalidArgument(
        "TLS provider rejected protocol version range");
  }
  return {};
}

Status encode_alpn(std::span<const container::StringView> protocols,
                   ContextImpl *context) noexcept {
  if (protocols.empty())
    return {};
  std::size_t encoded_size = 0;
  for (const container::StringView protocol : protocols) {
    if (protocol.data() == nullptr || protocol.empty() ||
        protocol.size() > 255) {
      return Status::InvalidArgument(
          "each TLS ALPN protocol must contain 1..255 bytes");
    }
    if (encoded_size >
        static_cast<std::size_t>((std::numeric_limits<unsigned int>::max)()) -
            protocol.size() - 1) {
      return Status::OutOfRange("TLS ALPN protocol list is too large");
    }
    encoded_size += static_cast<std::size_t>(protocol.size()) + 1;
  }
  auto *encoded = new (std::nothrow) unsigned char[encoded_size];
  if (encoded == nullptr) {
    return Status::ResourceExhausted("TLS ALPN allocation failed");
  }
  std::size_t offset = 0;
  for (const container::StringView protocol : protocols) {
    encoded[offset++] = static_cast<unsigned char>(protocol.size());
    std::memcpy(encoded + offset, protocol.data(), protocol.size());
    offset += protocol.size();
  }
  context->alpn = encoded;
  context->alpn_size = static_cast<unsigned int>(encoded_size);
  return {};
}

int select_alpn(SSL *, const unsigned char **out, unsigned char *out_size,
                const unsigned char *peer, unsigned int peer_size,
                void *argument) noexcept {
  auto *context = static_cast<ContextImpl *>(argument);
  if (context == nullptr || context->alpn_size == 0) {
    return SSL_TLSEXT_ERR_NOACK;
  }
  const int selected =
      SSL_select_next_proto(const_cast<unsigned char **>(out), out_size,
                            context->alpn, context->alpn_size, peer, peer_size);
  if (selected == OPENSSL_NPN_NEGOTIATED)
    return SSL_TLSEXT_ERR_OK;
  return context->require_alpn ? SSL_TLSEXT_ERR_ALERT_FATAL
                               : SSL_TLSEXT_ERR_NOACK;
}

Status finish_context(ContextImpl *context) noexcept {
  if (context->alpn_size != 0) {
    if (context->server) {
      SSL_CTX_set_alpn_select_cb(context->provider, select_alpn, context);
    } else if (SSL_CTX_set_alpn_protos(context->provider, context->alpn,
                                       context->alpn_size) != 0) {
      return Status::InvalidArgument("TLS provider rejected ALPN protocols");
    }
  } else if (context->require_alpn) {
    return Status::InvalidArgument(
        "TLS ALPN cannot be required without configured protocols");
  }
  return {};
}

enum class ProviderWant : std::uint8_t {
  kDone,
  kRead,
  kWrite,
  kCleanEof,
  kError,
};

ProviderWant classify(SSL *provider, int result) noexcept {
  switch (SSL_get_error(provider, result)) {
  case SSL_ERROR_NONE:
    return ProviderWant::kDone;
  case SSL_ERROR_WANT_READ:
    return ProviderWant::kRead;
  case SSL_ERROR_WANT_WRITE:
    return ProviderWant::kWrite;
  case SSL_ERROR_ZERO_RETURN:
    return ProviderWant::kCleanEof;
  default:
    return ProviderWant::kError;
  }
}

Status provider_failure(const char *operation) noexcept {
  // Portable recovery uses the canonical code. OpenSSL's thread-local numeric
  // queue is deliberately drained here rather than exported as a fake stable
  // native domain; callers may add tracing at the operation boundary.
  while (ERR_get_error() != 0) {
  }
  return Status::DataLoss(operation);
}

struct FeedResult {
  Status status{};
  bool eof = false;
};

Status operation_budget_status(const OperationOptions &options) noexcept {
  // Match io_core's BudgetState ordering before the optimistic transport seam
  // performs any syscall. Otherwise a socket that happens to be immediately
  // writable could leak protocol bytes after a pre-cancelled or already
  // expired operation should have terminated.
  if (options.cancel != nullptr && options.cancel->is_cancelled())
    return Status::Cancelled("TLS operation cancelled");
  if (options.deadline.has_value() &&
      *options.deadline <= time::monotonic_now()) {
    return Status::DeadlineExceeded("TLS operation deadline exceeded");
  }
  return {};
}

Status consume_sent_ciphertext(BIO *write_bio, std::byte *scratch,
                               std::size_t bytes) noexcept {
  while (bytes != 0) {
    const std::size_t request = (std::min)(bytes, kPumpBytes);
    const int consumed =
        BIO_read(write_bio, scratch, static_cast<int>(request));
    if (consumed <= 0 || static_cast<std::size_t>(consumed) != request) {
      return Status::Internal(
          "TLS provider lost pending ciphertext after transport write");
    }
    bytes -= request;
  }
  return {};
}

Status retain_received_ciphertext(BIO *read_bio, std::byte *scratch,
                                  const TransferResult &received) noexcept {
  if (received.bytes != 0) {
    const int accepted =
        BIO_write(read_bio, scratch, static_cast<int>(received.bytes));
    if (accepted != static_cast<int>(received.bytes)) {
      return Status::ResourceExhausted(
          "TLS provider input buffer rejected ciphertext");
    }
  }
  if (!received.status.ok())
    return received.status;
  if (received.eof) {
    return Status::DataLoss(
        "TLS transport ended while ciphertext output was pending");
  }
  if (received.bytes == 0) {
    return Status::Unavailable("TLS transport read made no forward progress");
  }
  return {};
}

sched::Task<Status> flush_ciphertext(mem::AllocatorRef, TcpStream &socket,
                                     BIO *read_bio, BIO *write_bio,
                                     std::byte *scratch,
                                     OperationOptions options) noexcept {
  bool opportunistic_read_used = false;
  for (;;) {
    Status budget = operation_budget_status(options);
    if (!budget.ok())
      co_return budget;
    char *pending = nullptr;
    const long available = BIO_get_mem_data(write_bio, &pending);
    if (available <= 0)
      co_return Status{};
    if (pending == nullptr) {
      co_return Status::Internal(
          "TLS provider reported pending ciphertext without storage");
    }
    const std::size_t pending_size =
        (std::min)(static_cast<std::size_t>(available), kPumpBytes);
    const auto output = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(pending), pending_size);

    detail::ImmediateTransferResult immediate =
        detail::tcp_try_write_some(socket, output);
    if (immediate.state == detail::ImmediateTransferState::kReady) {
      if (immediate.transfer.bytes != 0) {
        Status consumed = consume_sent_ciphertext(write_bio, scratch,
                                                  immediate.transfer.bytes);
        if (!consumed.ok())
          co_return consumed;
      }
      if (!immediate.transfer.status.ok())
        co_return immediate.transfer.status;
      if (immediate.transfer.bytes == 0)
        co_return Status::Unavailable(
            "TLS transport write made no forward progress");
      // A successful transport write ends the current pressure episode. A
      // later would-block may again consume one already-arrived peer record;
      // every reset is therefore backed by real output progress and cannot
      // create a read/write spin loop.
      opportunistic_read_used = false;
      continue;
    }
    if (immediate.state == detail::ImmediateTransferState::kUnsupported) {
      co_return Status::Internal(
          "TLS TCP transport does not support immediate progress");
    }

    const std::size_t retained_ciphertext = BIO_ctrl_pending(read_bio);
    if (!opportunistic_read_used &&
        retained_ciphertext < kMaxRetainedCiphertextBytes) {
      const std::size_t receive_capacity =
          (std::min)(kPumpBytes,
                     kMaxRetainedCiphertextBytes - retained_ciphertext);
      detail::ImmediateTransferResult received = detail::tcp_try_read_some(
          socket, std::span<std::byte>(scratch, receive_capacity));
      if (received.state == detail::ImmediateTransferState::kReady) {
        Status retained =
            retain_received_ciphertext(read_bio, scratch, received.transfer);
        if (!retained.ok())
          co_return retained;
        opportunistic_read_used = true;
        continue;
      }
      if (received.state == detail::ImmediateTransferState::kUnsupported) {
        co_return Status::Internal(
            "TLS TCP transport does not support immediate progress");
      }
    }

    // The pointer belongs to the memory BIO and remains stable while this
    // single-owner TLS operation is suspended: no provider/BIO call can run on
    // the stream until the await completes. Consume only bytes the socket has
    // actually accepted.
    TransferResult sent = co_await socket.write_some(output, options);
    if (sent.bytes != 0) {
      Status consumed = consume_sent_ciphertext(write_bio, scratch, sent.bytes);
      if (!consumed.ok())
        co_return consumed;
    }
    if (!sent.status.ok())
      co_return sent.status;
    if (sent.bytes == 0)
      co_return Status::Unavailable(
          "TLS transport write made no forward progress");
    opportunistic_read_used = false;
  }
}

sched::Task<FeedResult> feed_ciphertext(mem::AllocatorRef, TcpStream &socket,
                                        BIO *read_bio, std::byte *scratch,
                                        OperationOptions options) noexcept {
  TransferResult received = co_await socket.read_some(
      std::span<std::byte>(scratch, kPumpBytes), options);
  if (!received.status.ok())
    co_return FeedResult{received.status, false};
  // TransferResult deliberately permits bytes+EOF. Feed those final bytes to
  // OpenSSL first; a following read observes the transport EOF after the
  // provider has consumed all complete records/close_notify they contain.
  if (received.bytes == 0 && received.eof)
    co_return FeedResult{{}, true};
  if (received.bytes == 0) {
    co_return FeedResult{
        Status::Unavailable("TLS transport read made no forward progress"),
        false};
  }
  const int accepted =
      BIO_write(read_bio, scratch, static_cast<int>(received.bytes));
  if (accepted != static_cast<int>(received.bytes)) {
    co_return FeedResult{Status::ResourceExhausted(
                             "TLS provider input buffer rejected ciphertext"),
                         false};
  }
  co_return FeedResult{};
}

Status transport_eof_status(const ContextImpl *context,
                            const char *operation) noexcept {
  return context->allow_truncated_eof ? Status{} : Status::DataLoss(operation);
}

} // namespace

Result<TlsStream> TlsStream::make_(TcpStream socket, const TlsContext &owner,
                                   bool server,
                                   const TlsConnectOptions *options) noexcept {
  if (!socket.valid()) {
    return make_unexpected(
        Status::FailedPrecondition("TLS requires an open TCP stream"));
  }
  auto *context = as_context(owner.impl_);
  if (context == nullptr || context->provider == nullptr) {
    return make_unexpected(
        Status::FailedPrecondition("TLS context is not initialized"));
  }
  if (context->server != server) {
    return make_unexpected(Status::InvalidArgument(
        server ? "TLS accept requires a server context"
               : "TLS connect requires a client context"));
  }

  ERR_clear_error();
  SSL *provider = SSL_new(context->provider);
  if (provider == nullptr) {
    return make_unexpected(
        Status::ResourceExhausted("TLS stream allocation failed"));
  }
  BIO *read_bio = BIO_new(BIO_s_mem());
  BIO *write_bio = BIO_new(BIO_s_mem());
  if (read_bio == nullptr || write_bio == nullptr) {
    BIO_free(read_bio);
    BIO_free(write_bio);
    SSL_free(provider);
    return make_unexpected(
        Status::ResourceExhausted("TLS BIO allocation failed"));
  }
  SSL_set_bio(provider, read_bio, write_bio);
  SSL_set_mode(provider, SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER |
                             SSL_MODE_ENABLE_PARTIAL_WRITE);

  if (server) {
    SSL_set_accept_state(provider);
  } else {
    const TlsConnectOptions &connect_options = *options;
    OwnedCString server_name;
    const bool name_required = context->verify_peer;
    Status copied =
        server_name.assign(connect_options.server_name, name_required,
                           "verifying TLS requires server_name");
    if (!copied.ok()) {
      SSL_free(provider);
      return make_unexpected(std::move(copied));
    }
    if (server_name.get() != nullptr) {
      bool numeric = false;
      bool digits_and_dots = true;
      for (const char *p = server_name.get(); *p != '\0'; ++p) {
        if (*p == ':')
          numeric = true;
        if ((*p < '0' || *p > '9') && *p != '.')
          digits_and_dots = false;
      }
      numeric = numeric || digits_and_dots;
      int configured = 1;
      if (context->verify_peer) {
        configured = numeric ? X509_VERIFY_PARAM_set1_ip_asc(
                                   SSL_get0_param(provider), server_name.get())
                             : SSL_set1_host(provider, server_name.get());
      }
      if (configured == 1 && !numeric) {
        configured = SSL_set_tlsext_host_name(provider, server_name.get());
      }
      if (configured != 1) {
        SSL_free(provider);
        return make_unexpected(Status::InvalidArgument(
            "TLS provider rejected server identity configuration"));
      }
    }
    if (connect_options.resume_session != nullptr) {
      if (!connect_options.resume_session->valid()) {
        SSL_free(provider);
        return make_unexpected(
            Status::InvalidArgument("TLS resume session is empty"));
      }
      if (connect_options.resume_session->context_id_ != context->identity) {
        SSL_free(provider);
        return make_unexpected(Status::InvalidArgument(
            "TLS resume session belongs to a different context"));
      }
      if (SSL_set_session(provider,
                          static_cast<SSL_SESSION *>(
                              connect_options.resume_session->session_)) != 1) {
        SSL_free(provider);
        return make_unexpected(
            Status::InvalidArgument("TLS provider rejected resume session"));
      }
    }
    SSL_set_connect_state(provider);
  }

  std::byte *scratch =
      static_cast<std::byte *>(context->connection_allocator.allocate(
          kPumpBytes, alignof(std::max_align_t)));
  if (scratch == nullptr) {
    SSL_free(provider);
    return make_unexpected(
        Status::ResourceExhausted("TLS pump scratch allocation failed"));
  }

  retain_context(context);
  TlsStream stream;
  stream.socket_ = std::move(socket);
  stream.ssl_ = provider;
  stream.read_bio_ = read_bio;
  stream.write_bio_ = write_bio;
  stream.context_impl_ = context;
  stream.scratch_ = scratch;
  stream.scratch_allocator_ = context->connection_allocator;
  stream.server_ = server;
  return stream;
}

std::span<const container::StringView>
tls_http2_protocols(bool allow_http1) noexcept {
  static constexpr container::StringView h2_only[] = {{"h2", 2, true}};
  static constexpr container::StringView h2_and_http1[] = {
      {"h2", 2, true}, {"http/1.1", 8, true}};
  return allow_http1 ? std::span<const container::StringView>(h2_and_http1)
                     : std::span<const container::StringView>(h2_only);
}

TlsSession::~TlsSession() {
  SSL_SESSION_free(static_cast<SSL_SESSION *>(session_));
}

TlsSession::TlsSession(TlsSession &&other) noexcept
    : session_(std::exchange(other.session_, nullptr)),
      context_id_(std::exchange(other.context_id_, 0)) {}

TlsSession &TlsSession::operator=(TlsSession &&other) noexcept {
  if (this != &other) {
    SSL_SESSION_free(static_cast<SSL_SESSION *>(session_));
    session_ = std::exchange(other.session_, nullptr);
    context_id_ = std::exchange(other.context_id_, 0);
  }
  return *this;
}

TlsContext::~TlsContext() { release_context(as_context(impl_)); }

TlsContext::TlsContext(TlsContext &&other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

TlsContext &TlsContext::operator=(TlsContext &&other) noexcept {
  if (this != &other) {
    release_context(as_context(impl_));
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

Result<TlsContext>
TlsContext::make_client(const TlsClientOptions &options) noexcept {
  Status status = validate_client_credentials(options);
  if (!status.ok())
    return make_unexpected(std::move(status));
  const bool has_identity = options.certificate_chain.supplied();

  ERR_clear_error();
  SSL_CTX *provider = SSL_CTX_new(TLS_client_method());
  if (provider == nullptr) {
    return make_unexpected(
        Status::ResourceExhausted("TLS client context allocation failed"));
  }
  auto *context = new (std::nothrow) ContextImpl;
  if (context == nullptr) {
    SSL_CTX_free(provider);
    return make_unexpected(
        Status::ResourceExhausted("TLS context metadata allocation failed"));
  }
  context->provider = provider;
  context->identity =
      next_context_identity.fetch_add(1, std::memory_order_relaxed);
  context->verify_peer = options.verify == TlsVerifyMode::kPeer;
  context->require_alpn = options.require_alpn;
  context->allow_truncated_eof = options.allow_truncated_eof;
  context->connection_allocator = options.connection_allocator;

  if (!context->connection_allocator.valid()) {
    release_context(context);
    return make_unexpected(
        Status::InvalidArgument("TLS connection allocator is invalid"));
  }

  status =
      configure_versions(provider, options.min_version, options.max_version);
  if (status.ok() && has_identity)
    status = load_identity(provider, options.certificate_chain,
                           options.private_key, options.private_key_password);
  if (status.ok())
    status = encode_alpn(options.alpn_protocols, context);
  if (status.ok())
    status = finish_context(context);
  if (status.ok()) {
    SSL_CTX_set_verify(provider,
                       context->verify_peer ? SSL_VERIFY_PEER : SSL_VERIFY_NONE,
                       nullptr);
    if (context->verify_peer) {
      if (!options.certificate_authority.supplied()) {
        if (SSL_CTX_set_default_verify_paths(provider) != 1) {
          status = Status::FailedPrecondition(
              "TLS system trust store could not be loaded");
        }
      } else {
        status = load_certificate_authorities(provider,
                                              options.certificate_authority);
      }
    }
  }
  if (status.ok()) {
    SSL_CTX_set_session_cache_mode(provider, options.enable_session_resumption
                                                 ? SSL_SESS_CACHE_CLIENT
                                                 : SSL_SESS_CACHE_OFF);
    if (!options.enable_session_resumption) {
      SSL_CTX_set_options(provider, SSL_OP_NO_TICKET);
    }
  }
  if (!status.ok()) {
    release_context(context);
    return make_unexpected(std::move(status));
  }
  return TlsContext(context);
}

Result<TlsContext>
TlsContext::make_server(const TlsServerOptions &options) noexcept {
  Status status = validate_server_credentials(options);
  if (!status.ok())
    return make_unexpected(std::move(status));

  ERR_clear_error();
  SSL_CTX *provider = SSL_CTX_new(TLS_server_method());
  if (provider == nullptr) {
    return make_unexpected(
        Status::ResourceExhausted("TLS server context allocation failed"));
  }
  auto *context = new (std::nothrow) ContextImpl;
  if (context == nullptr) {
    SSL_CTX_free(provider);
    return make_unexpected(
        Status::ResourceExhausted("TLS context metadata allocation failed"));
  }
  context->provider = provider;
  context->identity =
      next_context_identity.fetch_add(1, std::memory_order_relaxed);
  context->server = true;
  context->verify_peer = options.client_verification == TlsVerifyMode::kPeer;
  context->require_alpn = options.require_alpn;
  context->allow_truncated_eof = options.allow_truncated_eof;
  context->connection_allocator = options.connection_allocator;

  if (!context->connection_allocator.valid()) {
    release_context(context);
    return make_unexpected(
        Status::InvalidArgument("TLS connection allocator is invalid"));
  }

  status =
      configure_versions(provider, options.min_version, options.max_version);
  if (status.ok())
    status = load_identity(provider, options.certificate_chain,
                           options.private_key, options.private_key_password);
  if (status.ok())
    status = encode_alpn(options.alpn_protocols, context);
  if (status.ok())
    status = finish_context(context);
  if (status.ok()) {
    SSL_CTX_set_verify(provider,
                       context->verify_peer
                           ? SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT
                           : SSL_VERIFY_NONE,
                       nullptr);
    if (context->verify_peer) {
      status = load_certificate_authorities(
          provider, options.client_certificate_authority);
    }
  }
  if (status.ok()) {
    SSL_CTX_set_session_cache_mode(provider, options.enable_session_resumption
                                                 ? SSL_SESS_CACHE_SERVER
                                                 : SSL_SESS_CACHE_OFF);
    if (options.enable_session_resumption) {
      const auto *identity =
          reinterpret_cast<const unsigned char *>(&context->identity);
      if (SSL_CTX_set_session_id_context(provider, identity,
                                         sizeof(context->identity)) != 1) {
        status = Status::Internal("TLS session cache identity setup failed");
      }
    } else {
      SSL_CTX_set_options(provider, SSL_OP_NO_TICKET);
    }
  }
  if (!status.ok()) {
    release_context(context);
    return make_unexpected(std::move(status));
  }
  return TlsContext(context);
}

TlsStream::~TlsStream() { reset_(); }

TlsStream::TlsStream(TlsStream &&other) noexcept
    : socket_(std::move(other.socket_)),
      ssl_(std::exchange(other.ssl_, nullptr)),
      read_bio_(std::exchange(other.read_bio_, nullptr)),
      write_bio_(std::exchange(other.write_bio_, nullptr)),
      context_impl_(std::exchange(other.context_impl_, nullptr)),
      scratch_(std::exchange(other.scratch_, nullptr)),
      scratch_allocator_(other.scratch_allocator_), server_(other.server_),
      handshake_complete_(other.handshake_complete_) {}

TlsStream &TlsStream::operator=(TlsStream &&other) noexcept {
  if (this != &other) {
    reset_();
    socket_ = std::move(other.socket_);
    ssl_ = std::exchange(other.ssl_, nullptr);
    read_bio_ = std::exchange(other.read_bio_, nullptr);
    write_bio_ = std::exchange(other.write_bio_, nullptr);
    context_impl_ = std::exchange(other.context_impl_, nullptr);
    scratch_ = std::exchange(other.scratch_, nullptr);
    scratch_allocator_ = other.scratch_allocator_;
    server_ = other.server_;
    handshake_complete_ = other.handshake_complete_;
  }
  return *this;
}

void TlsStream::reset_() noexcept {
  SSL_free(as_ssl(ssl_));
  ssl_ = nullptr;
  read_bio_ = nullptr;
  write_bio_ = nullptr;
  release_context(as_context(context_impl_));
  context_impl_ = nullptr;
  if (scratch_ != nullptr) {
    scratch_allocator_.deallocate(scratch_, kPumpBytes,
                                  alignof(std::max_align_t));
    scratch_ = nullptr;
  }
  handshake_complete_ = false;
}

Result<TlsStream>
TlsStream::connect(TcpStream socket, const TlsContext &context,
                   const TlsConnectOptions &options) noexcept {
  return make_(std::move(socket), context, false, &options);
}

Result<TlsStream> TlsStream::accept(TcpStream socket,
                                    const TlsContext &context) noexcept {
  return make_(std::move(socket), context, true, nullptr);
}

container::StringView TlsStream::negotiated_protocol() const noexcept {
  if (ssl_ == nullptr || !handshake_complete_)
    return {};
  const unsigned char *selected = nullptr;
  unsigned int selected_size = 0;
  SSL_get0_alpn_selected(as_ssl(ssl_), &selected, &selected_size);
  return container::StringView(reinterpret_cast<const char *>(selected),
                               selected_size);
}

bool TlsStream::session_reused() const noexcept {
  return ssl_ != nullptr && handshake_complete_ &&
         SSL_session_reused(as_ssl(ssl_)) == 1;
}

Result<TlsSession> TlsStream::take_session() const noexcept {
  if (ssl_ == nullptr || !handshake_complete_ || server_) {
    return make_unexpected(Status::FailedPrecondition(
        "TLS client handshake must complete before taking a session"));
  }
  SSL_SESSION *session = SSL_get1_session(as_ssl(ssl_));
  if (session == nullptr || SSL_SESSION_is_resumable(session) != 1) {
    SSL_SESSION_free(session);
    return make_unexpected(
        Status::FailedPrecondition("TLS provider has no resumable session"));
  }
  return TlsSession(session, as_context(context_impl_)->identity);
}

sched::Task<Status> tls_handshake(mem::AllocatorRef allocator,
                                  TlsStream &stream,
                                  OperationOptions options) noexcept {
  if (!stream.valid()) {
    co_return Status::FailedPrecondition("TLS stream is not initialized");
  }
  if (stream.handshake_complete_)
    co_return Status{};
  std::byte *scratch = stream.scratch_;
  SSL *provider = as_ssl(stream.ssl_);
  for (;;) {
    ERR_clear_error();
    const int result = SSL_do_handshake(provider);
    const ProviderWant want = classify(provider, result);
    Status flushed = co_await flush_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_),
        as_bio(stream.write_bio_), scratch, options);
    if (!flushed.ok())
      co_return flushed;
    if (want == ProviderWant::kDone) {
      auto *context = as_context(stream.context_impl_);
      Status checked;
      if (context->verify_peer &&
          SSL_get_verify_result(provider) != X509_V_OK) {
        checked =
            Status::Unauthenticated("TLS peer certificate verification failed");
      } else if (context->require_alpn) {
        const unsigned char *selected = nullptr;
        unsigned int selected_size = 0;
        SSL_get0_alpn_selected(provider, &selected, &selected_size);
        if (selected_size == 0) {
          checked = Status::FailedPrecondition(
              "TLS peer did not negotiate a required ALPN protocol");
        }
      }
      if (checked.ok())
        stream.handshake_complete_ = true;
      co_return checked;
    }
    if (want == ProviderWant::kCleanEof) {
      co_return Status::DataLoss("TLS peer closed during handshake");
    }
    if (want == ProviderWant::kError) {
      auto *context = as_context(stream.context_impl_);
      if (context->verify_peer &&
          SSL_get_verify_result(provider) != X509_V_OK) {
        while (ERR_get_error() != 0) {
        }
        co_return Status::Unauthenticated(
            "TLS peer certificate verification failed");
      }
      if (context->require_alpn) {
        const unsigned char *selected = nullptr;
        unsigned int selected_size = 0;
        SSL_get0_alpn_selected(provider, &selected, &selected_size);
        if (selected_size == 0) {
          while (ERR_get_error() != 0) {
          }
          co_return Status::FailedPrecondition(
              "TLS peer did not negotiate a required ALPN protocol");
        }
      }
      co_return provider_failure("TLS handshake failed");
    }
    if (want == ProviderWant::kWrite)
      continue;
    FeedResult fed = co_await feed_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_), scratch, options);
    if (!fed.status.ok())
      co_return fed.status;
    if (fed.eof)
      co_return Status::DataLoss("truncated TLS handshake");
  }
}

sched::Task<TransferResult> tls_read(mem::AllocatorRef allocator,
                                     TlsStream &stream,
                                     std::span<std::byte> buffer,
                                     OperationOptions options) noexcept {
  if (!stream.valid() || !stream.handshake_complete_) {
    co_return TransferResult{
        0, Status::FailedPrecondition("TLS handshake is not complete"), false};
  }
  if (buffer.empty())
    co_return TransferResult{};
  std::byte *scratch = stream.scratch_;
  SSL *provider = as_ssl(stream.ssl_);
  for (;;) {
    std::size_t bytes = 0;
    ERR_clear_error();
    const int result =
        SSL_read_ex(provider, buffer.data(), buffer.size(), &bytes);
    const ProviderWant want = classify(provider, result);
    Status flushed = co_await flush_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_),
        as_bio(stream.write_bio_), scratch, options);
    if (!flushed.ok())
      co_return TransferResult{bytes, flushed, false};
    if (result == 1 && bytes != 0) {
      co_return TransferResult{bytes, {}, false};
    }
    if (want == ProviderWant::kCleanEof) {
      co_return TransferResult{bytes, {}, true};
    }
    if (want == ProviderWant::kError) {
      co_return TransferResult{
          bytes, provider_failure("TLS record read failed"), false};
    }
    if (want == ProviderWant::kWrite)
      continue;
    FeedResult fed = co_await feed_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_), scratch, options);
    if (!fed.status.ok()) {
      co_return TransferResult{bytes, fed.status, false};
    }
    if (fed.eof) {
      Status eof_status =
          transport_eof_status(as_context(stream.context_impl_),
                               "TLS transport ended without close_notify");
      co_return TransferResult{bytes, eof_status, eof_status.ok()};
    }
  }
}

sched::Task<TransferResult> tls_write(mem::AllocatorRef allocator,
                                      TlsStream &stream,
                                      std::span<const std::byte> buffer,
                                      OperationOptions options) noexcept {
  if (!stream.valid() || !stream.handshake_complete_) {
    co_return TransferResult{
        0, Status::FailedPrecondition("TLS handshake is not complete"), false};
  }
  if (buffer.empty())
    co_return TransferResult{};
  std::byte *scratch = stream.scratch_;
  SSL *provider = as_ssl(stream.ssl_);
  std::size_t completed = 0;
  for (;;) {
    std::size_t bytes = 0;
    ERR_clear_error();
    const int result = SSL_write_ex(provider, buffer.data() + completed,
                                    buffer.size() - completed, &bytes);
    const ProviderWant want = classify(provider, result);
    completed += bytes;
    Status flushed = co_await flush_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_),
        as_bio(stream.write_bio_), scratch, options);
    if (!flushed.ok())
      co_return TransferResult{completed, flushed, false};
    if (result == 1) {
      if (bytes == 0) {
        co_return TransferResult{
            completed,
            Status::Unavailable("TLS write made no forward progress"), false};
      }
      // SSL_MODE_ENABLE_PARTIAL_WRITE makes one successful provider call one
      // record-sized write-some operation. Returning here keeps peer input
      // retained by the duplex progress pump bounded; write-all belongs to the
      // caller/composed transport and already loops TransferResult progress.
      co_return TransferResult{completed, {}, false};
    }
    if (want == ProviderWant::kCleanEof) {
      co_return TransferResult{
          completed, Status::Unavailable("TLS peer closed during write"),
          false};
    }
    if (want == ProviderWant::kError) {
      co_return TransferResult{
          completed, provider_failure("TLS record write failed"), false};
    }
    if (want == ProviderWant::kWrite)
      continue;
    FeedResult fed = co_await feed_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_), scratch, options);
    if (!fed.status.ok()) {
      co_return TransferResult{completed, fed.status, false};
    }
    if (fed.eof) {
      co_return TransferResult{
          completed, Status::DataLoss("TLS peer closed during record write"),
          false};
    }
  }
}

sched::Task<Status> tls_shutdown(mem::AllocatorRef allocator, TlsStream &stream,
                                 TlsShutdownOptions options) noexcept {
  if (!stream.valid())
    co_return Status{};
  if (!stream.handshake_complete_) {
    co_return Status::FailedPrecondition("TLS handshake is not complete");
  }
  std::byte *scratch = stream.scratch_;
  SSL *provider = as_ssl(stream.ssl_);
  for (;;) {
    ERR_clear_error();
    const int result = SSL_shutdown(provider);
    ProviderWant want = ProviderWant::kDone;
    if (result < 0)
      want = classify(provider, result);
    Status flushed = co_await flush_ciphertext(
        allocator, stream.socket_, as_bio(stream.read_bio_),
        as_bio(stream.write_bio_), scratch, options.operation);
    if (!flushed.ok())
      co_return flushed;
    if (result == 1 ||
        (result == 0 && options.mode == TlsShutdownMode::kSendOnly)) {
      co_return Status{};
    }
    if (result < 0 && want == ProviderWant::kError) {
      co_return provider_failure("TLS shutdown failed");
    }
    if (result < 0 && want == ProviderWant::kCleanEof)
      co_return Status{};
    if (result < 0 && want == ProviderWant::kWrite)
      continue;

    FeedResult fed = co_await feed_ciphertext(allocator, stream.socket_,
                                              as_bio(stream.read_bio_), scratch,
                                              options.operation);
    if (!fed.status.ok())
      co_return fed.status;
    if (fed.eof) {
      co_return transport_eof_status(
          as_context(stream.context_impl_),
          "TLS transport ended before peer close_notify");
    }
  }
}

} // namespace hyle::io

#endif // HYLE_IO_TLS
