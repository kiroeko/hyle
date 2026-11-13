"""Immutable cold choices and per-request selection; no model or HTTP code."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path


@dataclass(frozen=True)
class ExecutionChoice:
    name: str = 'default'
    drafts: int | None = None
    overlap: bool | None = None
    assets: tuple[tuple[str, Path], ...] = ()
    generate: dict | None = None


@dataclass(frozen=True)
class SamplingChoice:
    name: str = 'default'
    random: bool = False
    temperature: float = 1.
    top_k: int = 0
    top_p: float = 1.


@dataclass(frozen=True)
class Selection:
    execution_profile: int = 0
    sampling_index: int = 0
    seed: int | None = None
    max_draft_tokens: int | None = None

    def worker_options(self) -> dict:
        return {key: value for key, value in (
            ('execution_profile', self.execution_profile or None), ('sampling_index', self.sampling_index or None),
            ('seed', self.seed), ('max_draft_tokens', self.max_draft_tokens)) if value is not None}


@dataclass(frozen=True)
class SessionPolicy:
    capacity: int = 128
    kv_capacity: int = 2
    ttl_seconds: float = 900.
    max_waiting: int = 64
    shared_prefix: bool = False


@dataclass(frozen=True)
class PreparedChoices:
    executions: tuple[ExecutionChoice, ...] = (ExecutionChoice(),)
    samplers: tuple[SamplingChoice, ...] = (SamplingChoice(),)
    sessions: SessionPolicy = SessionPolicy()
    prefix_entries: int = 0

    def select(self, values=None, *, temperature=None, top_p=None, top_k=None) -> Selection:
        values = {} if values is None else values
        if not isinstance(values, dict) or set(values) - {'execution_profile', 'sampling_profile', 'seed', 'max_draft_tokens'}:
            raise ValueError('unsupported hyle request selection')
        def named(key, choices):
            name = values.get(key, 'default')
            if not isinstance(name, str):
                raise ValueError(key + ' must name a cold-prepared choice')
            for index, choice in enumerate(choices):
                if choice.name == name:
                    return index
            raise ValueError(key + ' was not cold prepared: ' + name)
        execution = named('execution_profile', self.executions)
        sampling = named('sampling_profile', self.samplers)
        if temperature is not None or top_p is not None or top_k is not None:
            for value, label in ((temperature, 'temperature'), (top_p, 'top_p')):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(label + ' must be finite')
            if temperature is not None and temperature < 0 or top_p is not None and not 0 < top_p <= 1:
                raise ValueError('sampling parameters exceed their supported range')
            if top_k is not None and (type(top_k) is not int or top_k < 0):
                raise ValueError('top_k must be a nonnegative integer')
            matches = [index for index, choice in enumerate(self.samplers)
                       if (temperature is None or (choice.temperature if choice.random else 0.) == temperature)
                       and (top_p is None or choice.top_p == top_p)
                       and (top_k is None or choice.top_k == top_k)]
            if 'sampling_profile' in values:
                if sampling not in matches:
                    raise ValueError('sampling parameters differ from the selected prepared sampler')
            elif matches:
                sampling = matches[0]
            else:
                raise ValueError('requested sampling program was not cold prepared')
        seed, drafts = values.get('seed'), values.get('max_draft_tokens')
        for value in (seed, drafts):
            if value is not None and (type(value) is not int or not 0 <= value < 1 << 64):
                raise ValueError('seed and draft ceiling must be unsigned 64-bit integers')
        limit = self.executions[execution].drafts
        if drafts is not None and limit is not None and drafts > limit:
            raise ValueError('draft ceiling exceeds its prepared execution profile')
        return Selection(execution, sampling, seed, drafts)


def read_configuration(args) -> PreparedChoices:
    data = {}
    if args.config:
        try:
            import tomllib
        except ModuleNotFoundError:
            try: import tomli as tomllib
            except ModuleNotFoundError as error:
                raise ValueError('TOML configuration requires tomllib or the optional tomli package') from error
        data = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if set(data) - {'default_profile', 'profiles', 'samplers', 'sessions', 'prefix_cache'}:
        raise ValueError('unknown text configuration section')
    directory = args.config.resolve().parent if args.config else Path.cwd()
    def execution(value, *, default=False):
        if not isinstance(value, dict) or set(value) - {'name', 'draft_count', 'overlap', 'assets', 'generate'}:
            raise ValueError('invalid prepared execution profile')
        name = value.get('name', 'default' if default else '')
        drafts = value.get('draft_count', args.draft_count if default else None)
        overlap = value.get('overlap', args.overlap if default else None)
        if not isinstance(name, str) or not name or drafts is not None and (type(drafts) is not int or drafts < 0):
            raise ValueError('execution profile requires a name and nonnegative draft count')
        if overlap is not None and type(overlap) is not bool:
            raise ValueError('overlap must be boolean')
        assets = value.get('assets', {})
        if not isinstance(assets, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in assets.items()):
            raise ValueError('profile assets must bind roles to local files')
        files = dict((k, (directory / v).resolve()) for k, v in assets.items())
        if default:
            if args.draft_vocabulary: files.setdefault('draft.vocabulary', args.draft_vocabulary.resolve())
            if args.draft_head: files.setdefault('draft.head', args.draft_head.resolve())
        recipe = value.get('generate')
        if recipe is not None:
            if not isinstance(recipe, dict):
                raise ValueError('asset generation requires a provider-owned recipe table')
            if files:
                raise ValueError('choose generated assets or explicit files for a profile')
        return ExecutionChoice(name, drafts, overlap, tuple(files.items()), recipe)
    executions = (execution(data.get('default_profile', {}), default=True),) + tuple(execution(x) for x in data.get('profiles', []))
    if executions[0].name != 'default':
        raise ValueError('the default execution profile is named default')
    samplers = [SamplingChoice('default', args.sampling == 'random', args.temperature, args.top_k, args.top_p)]
    for value in data.get('samplers', []):
        if not isinstance(value, dict) or set(value) - {'name', 'mode', 'temperature', 'top_k', 'top_p'}:
            raise ValueError('invalid prepared sampling profile')
        mode = value.get('mode', 'random')
        if mode not in ('greedy', 'random'):
            raise ValueError('sampler mode must be greedy or random')
        sampler = SamplingChoice(value.get('name', ''), mode == 'random', value.get('temperature', 1.),
                                  value.get('top_k', 0), value.get('top_p', 1.))
        if not isinstance(sampler.name, str) or not sampler.name or type(sampler.top_k) is not int or sampler.top_k < 0:
            raise ValueError('invalid sampler name or top-k')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
               for x in (sampler.temperature, sampler.top_p)) or sampler.top_p > 1:
            raise ValueError('invalid prepared sampler parameters')
        if not sampler.random and (sampler.temperature != 1 or sampler.top_p != 1 or sampler.top_k):
            raise ValueError('greedy sampler does not consume random parameters')
        samplers.append(sampler)
    for choices in (executions, samplers):
        if len({choice.name for choice in choices}) != len(choices):
            raise ValueError('duplicate prepared choice name')
    sessions = data.get('sessions', {})
    prefix = data.get('prefix_cache', {})
    if set(sessions) - {'capacity', 'kv_capacity', 'ttl_seconds', 'max_waiting'} or set(prefix) - {'entries', 'shared'}:
        raise ValueError('unknown session or prefix policy')
    policy = SessionPolicy(**sessions, shared_prefix=prefix.get('shared', False))
    if 'kv_capacity' not in sessions: policy = replace(policy, kv_capacity=args.max_sessions)
    if any(type(x) is not int or x <= 0 for x in (policy.capacity, policy.kv_capacity, policy.max_waiting)) or policy.kv_capacity > args.max_sessions:
        raise ValueError('session capacities must be positive and KV capacity must fit native residents')
    if isinstance(policy.ttl_seconds, bool) or not isinstance(policy.ttl_seconds, (int, float)) or not math.isfinite(policy.ttl_seconds) or policy.ttl_seconds <= 0:
        raise ValueError('session TTL must be finite and positive')
    entries = prefix.get('entries', 0)
    if type(entries) is not int or entries < 0 or type(policy.shared_prefix) is not bool:
        raise ValueError('invalid trusted prefix policy')
    return PreparedChoices(executions, tuple(samplers), policy, entries)
