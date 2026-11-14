"""Shared text preparation and delivery, independent of wire protocols."""
from __future__ import annotations

import threading
import json
import uuid
from array import array
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from .codec import TextCodec, has_images
from .contracts import TokenEvent, TokenRequest, TokenWorker, WorkerError, PromptAttachment, ToolOptions, ToolCall
from .configuration import PreparedChoices, Selection, SessionPolicy
from .sessions import TextSessions


@dataclass(frozen=True)
class PreparedText:
    ids: list[int]
    limit: int
    selection: Selection = Selection()
    attachments: tuple[PromptAttachment, ...] = ()
    tools: ToolOptions | None = None


@dataclass
class GenerationProgress:
    count: int = 0
    last: int | None = None
    stop_sequence: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class _TextStops:
    """Hold only a possible delimiter prefix; never publish part of a stop."""
    def __init__(self, sequences: Sequence[str]):
        self.sequences = tuple(sequences)
        if any(not isinstance(stop, str) or not stop for stop in self.sequences):
            raise ValueError('text stop sequences must be nonempty strings')
        self.pending = ''
        self.states = [0] * len(self.sequences)
        self.prefixes = []
        for stop in self.sequences:
            prefix = array('I', [0]) * len(stop)
            for index in range(1, len(stop)):
                length = prefix[index - 1]
                while length and stop[index] != stop[length]: length = prefix[length - 1]
                if stop[index] == stop[length]: length += 1
                prefix[index] = length
            self.prefixes.append(prefix)

    def push(self, chunk: str, *, final: bool = False) -> tuple[str, str | None]:
        text = self.pending + chunk
        for position, character in enumerate(chunk):
            matches = []
            for index, stop in enumerate(self.sequences):
                length, prefix = self.states[index], self.prefixes[index]
                while length and character != stop[length]: length = prefix[length - 1]
                if character == stop[length]: length += 1
                self.states[index] = length
                if length == len(stop): matches.append(stop)
            if matches:
                stop = max(matches, key=len)
                visible = text[:len(self.pending) + position + 1 - len(stop)]
                self.pending = ''
                return visible, stop
        keep = 0 if final else max(self.states, default=0)
        self.pending = text[-keep:] if keep else ''
        return text[:-keep] if keep else text, None


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = self.completed = self.failed = self.output_tokens = 0

    def begin(self) -> None:
        with self._lock:
            self.requests += 1

    def end(self, completed: bool, tokens: int) -> None:
        with self._lock:
            self.completed += int(completed)
            self.failed += int(not completed)
            self.output_tokens += tokens

    def render(self) -> bytes:
        with self._lock:
            return (f"hyle_requests_total {self.requests}\n"
                    f"hyle_requests_completed_total {self.completed}\n"
                    f"hyle_requests_failed_total {self.failed}\n"
                    f"hyle_output_tokens_total {self.output_tokens}\n").encode("ascii")


class TextEngine:
    def __init__(self, worker: TokenWorker, codec: TextCodec, model_name: str, *,
                 choices: PreparedChoices | None = None, session_policy: SessionPolicy | None = None):
        self.worker, self.codec, self.model_name = worker, codec, model_name
        self.stop_ids = frozenset(worker.stop_ids | codec.stop_ids)
        self.extra_stop_ids = tuple(sorted(self.stop_ids - worker.stop_ids))
        self.metrics = Metrics()
        self.choices = choices or PreparedChoices()
        self.execution_profiles = tuple(getattr(worker, 'execution_profiles', ()))
        if self.execution_profiles and tuple(profile.name for profile in self.execution_profiles) != tuple(
                profile.name for profile in self.choices.executions):
            raise ValueError('prepared execution profiles differ from the requested configuration')
        self.sessions = TextSessions(worker, session_policy, self.extra_stop_ids,
                                    getattr(codec, 'history_message', None)) if session_policy else None

    def prepare(self, messages: Sequence[dict[str, Any]], limit: int, selection: Selection = Selection(),
                previous_tape: Sequence[int] | None = None, *, continue_final_message: bool = False,
                tools: ToolOptions | None = None) -> PreparedText:
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        try:
            tool_encode = getattr(self.codec, 'encode_tool_messages', None) if tools is not None else None
            if tools is not None and tools.enabled and (
                    not callable(tool_encode) or not callable(getattr(self.codec, 'new_tool_decoder', None))):
                raise ValueError('the model codec does not support tool calls')
            attachments = ()
            images = has_images(messages)
            if images and not self.execution_profiles:
                raise ValueError('model worker did not report prepared image capabilities')
            if self.execution_profiles:
                modality = 'image' if images else 'text'
                selected = self.execution_profiles[selection.execution_profile]
                if modality not in selected.input_modalities:
                    matches = [i for i, profile in enumerate(self.execution_profiles)
                               if modality in profile.input_modalities]
                    if selection.execution_profile or len(matches) != 1:
                        raise ValueError(f'select a cold-prepared {modality} execution profile')
                    selection = replace(selection, execution_profile=matches[0])
            if self.execution_profiles and selection.max_draft_tokens is not None and (
                    selection.max_draft_tokens > self.execution_profiles[selection.execution_profile].max_draft_tokens):
                raise ValueError('draft ceiling exceeds the actual prepared execution profile')
            if images:
                prepare_images=getattr(self.codec,'prepare_images',None)
                if prepare_images is None: raise ValueError('the loaded codec has no image preparation recipe')
                prepared = prepare_images(messages, continue_final_message=continue_final_message,
                    max_tokens=self.worker.context-limit+1, **({'tools': tools} if tool_encode is not None else {}))
                ids, attachments = prepared.ids, prepared.attachments
            continuation = getattr(self.codec, 'continue_tool_messages' if tools is not None else 'continue_messages', None)
            if not images:
                ids = continuation(previous_tape, messages, **({'tools': tools} if tools is not None else {})) if continuation and previous_tape and not continue_final_message else None
            if ids is None:
                options = {'continue_final_message': True} if continue_final_message else {}
                if tool_encode is not None:
                    ids = tool_encode(messages, tools=tools, **options)
                else: ids = self.codec.encode_messages(messages, **options)
        except Exception as error:
            raise ValueError("chat template could not render the supplied messages: " + str(error)) from error
        if len(ids) > self.worker.context or limit - 1 > self.worker.context - len(ids):
            raise ValueError("prompt and output limit exceed the loaded model context")
        return PreparedText(ids, limit, selection, attachments, tools)

    def submit(self, prepared: PreparedText) -> TokenRequest:
        return self.worker.submit(prepared.ids, prepared.limit, stop_ids=self.extra_stop_ids,
                                  **prepared.selection.worker_options(),
                                  **({'attachments': prepared.attachments} if prepared.attachments else {}))

    def describe_capabilities(self) -> dict:
        describe = getattr(self.codec, 'describe_capabilities', None)
        frontend = describe() if describe is not None else {'input_modalities': None,
            'tools': {'supported': None if callable(getattr(self.codec, 'new_tool_decoder', None)) else False,
                      'strict': False}}
        return {'available': self.healthy, 'context_tokens': self.worker.context,
            'vocabulary_size': getattr(self.worker, 'vocabulary', None),
            'source_identity': getattr(self.worker, 'source_identity', None),
            'execution_profiles': [{'name': profile.name, 'algorithm': profile.algorithm,
                'max_draft_tokens': profile.max_draft_tokens, 'input_modalities': list(profile.input_modalities),
                'co_resident_with': [self.execution_profiles[i].name for i in profile.co_resident_with]}
                for profile in self.execution_profiles],
            'sampling_presets': [{'name': choice.name, 'mode': 'random' if choice.random else 'greedy',
                'temperature': choice.temperature if choice.random else 0., 'top_k': choice.top_k,
                'top_p': choice.top_p} for choice in self.choices.samplers],
            'sampling_selection': 'cold_configuration', 'codec': frontend,
            'structured_output': {'json_mode': False, 'strict_schema': False},
            'sessions': {'enabled': self.sessions is not None,
                **({'history_capacity': self.sessions.policy.capacity,
                    'resident_capacity': self.sessions.policy.kv_capacity,
                    'max_waiting': self.sessions.policy.max_waiting} if self.sessions is not None else {})}}

    def begin_close(self) -> None:
        if self.sessions: self.sessions.begin_close()

    @property
    def healthy(self) -> bool:
        return self.worker.healthy and (self.sessions is None or self.sessions.healthy)

    def close(self) -> None:
        if self.sessions: self.sessions.close()

    def generate(self, request: TokenRequest, on_text: Callable[[str], None], progress: GenerationProgress,
                 read: Callable[[], TokenEvent] | None = None, *, stop_sequences: Sequence[str] = (),
                 prepared: PreparedText | None = None, on_tool_call: Callable[[ToolCall], None] | None = None) -> str:
        chunks: list[str] = []
        tool_mode = prepared is not None and prepared.tools is not None and prepared.tools.enabled
        text_stops = _TextStops(stop_sequences) if stop_sequences else None
        stops = None if tool_mode else text_stops
        def tool_text(chunk):
            if chunk: on_text(chunk); chunks.append(chunk)
            return progress.stop_sequence is None
        def filter_tools(chunk, *, final=False):
            if text_stops: chunk, progress.stop_sequence = text_stops.push(chunk, final=final)
            return chunk
        def accept_call(name, arguments):
            call = ToolCall('call_' + uuid.uuid4().hex, name,
                json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            progress.tool_calls += (call,)
            if on_tool_call is not None: on_tool_call(call)
        if tool_mode:
            decoder = getattr(self.codec, 'new_tool_decoder')(prepared.tools, accept_call,
                lambda: progress.stop_sequence is not None or progress.count == prepared.limit and progress.last not in self.stop_ids,
                tool_text, filter_tools)
        else: decoder = self.codec.new_decoder()
        read_next = read if read is not None else request.read
        while True:
            event = read_next()
            if event.kind == "token":
                assert event.token is not None
                progress.count += 1
                progress.last = event.token
                chunk = "" if event.token in self.stop_ids else decoder.push(event.token)
                if stops: chunk, progress.stop_sequence = stops.push(chunk)
                if chunk:
                    on_text(chunk)
                    chunks.append(chunk)
                if progress.stop_sequence is not None: return ''.join(chunks)
                # The consumer must accept the text before returning credit.
                request.acknowledge()
            elif event.kind == "completed":
                tail = decoder.finish()
                if stops: tail, progress.stop_sequence = stops.push(tail, final=True)
                if tail:
                    on_text(tail)
                    chunks.append(tail)
                return "".join(chunks)
            elif event.kind == "error":
                raise WorkerError(event.error_kind or "generation", event.message)
            else:
                raise WorkerError("generation", "generation " + event.kind)
