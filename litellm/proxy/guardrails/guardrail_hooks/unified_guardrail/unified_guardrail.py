"""
Unified Guardrail, leveraging LiteLLM's /applyGuardrail endpoint

1. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_pre_call_hook
2. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_success_hook
3. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_streaming_iterator_hook
"""

import copy
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.cost_calculator import _infer_call_type
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.llms import load_guardrail_translation_mappings
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, CallTypesLiteral

# Call types that use NDJSON streaming (A2A); guardrail HTTPException is emitted as in-stream error
A2A_CALL_TYPES = (CallTypes.asend_message, CallTypes.send_message)

GUARDRAIL_NAME = "unified_llm_guardrails"

# Iterator-hook mode constants for streaming guardrails.
#
# `moderation` (default) preserves the historical behavior: the guardrail
# is sampled on accumulated text for compliance scanning / BLOCKED
# decisions, and the original upstream chunks are forwarded to the client
# verbatim. Anything the guardrail returns in `texts` is observed but
# not applied to the outbound stream.
#
# `transform` flips the yield strategy: the guardrail's modified texts
# are yielded to the client instead of the original upstream chunks. At
# each sample point the chunk content emitted is the new prefix of the
# guardrail's modified accumulated output (i.e. the part that hasn't
# been yielded yet); chunks between samples are buffered and drained at
# the next sample or at end-of-stream. Required for guardrails that
# need to rewrite (mask, redact, deanonymize, …) streamed content
# before it reaches the client.
#
# Opt-in per guardrail. Default is `moderation` so existing
# moderation-only consumers are unaffected.
ITERATOR_HOOK_MODE_MODERATION = "moderation"
ITERATOR_HOOK_MODE_TRANSFORM = "transform"
_VALID_ITERATOR_HOOK_MODES = (
    ITERATOR_HOOK_MODE_MODERATION,
    ITERATOR_HOOK_MODE_TRANSFORM,
)


def _get_a2a_request_id(
    responses_so_far: List[Any], request_data: dict
) -> Optional[str]:
    """Get JSON-RPC request id from first A2A chunk or request body for in-stream error reporting."""
    for item in responses_so_far:
        if isinstance(item, dict) and "id" in item:
            return item.get("id")
        if isinstance(item, str):
            try:
                obj = json.loads(item.strip())
                if isinstance(obj, dict) and "id" in obj:
                    return obj.get("id")
            except (json.JSONDecodeError, TypeError):
                continue
    body = request_data.get("body") or request_data.get("data") or {}
    if isinstance(body, dict):
        return body.get("id")
    return None


endpoint_guardrail_translation_mappings = None


# ── Transform-mode helpers ────────────────────────────────────────────


def _resolve_iterator_hook_mode(
    guardrail_to_apply: Any,
    optional_params: dict,
) -> str:
    """Resolve the iterator-hook mode for a streaming post_call.

    Mirrors the resolution chain `_resolve_streaming_*` uses for
    `streaming_sampling_rate` / `streaming_end_of_stream_only`:

      1. Direct attribute on the guardrail instance.
      2. `guardrail_config` dict on the guardrail instance.
      3. The hook's own `optional_params` (kwargs passed to
         `UnifiedLLMGuardrails.__init__`).

    Falls back to the moderation default when nothing matches. Unknown
    string values are coerced to the default with a warning so a
    config typo can't silently turn off transform mode for a guardrail
    the operator intended to enable it on.
    """
    mode: str = ITERATOR_HOOK_MODE_MODERATION

    if guardrail_to_apply is not None:
        mode = getattr(guardrail_to_apply, "iterator_hook_mode", mode)
        guardrail_config = getattr(guardrail_to_apply, "guardrail_config", {})
        if isinstance(guardrail_config, dict):
            mode = guardrail_config.get("iterator_hook_mode", mode)

    mode = optional_params.get("iterator_hook_mode", mode)

    if not isinstance(mode, str) or mode not in _VALID_ITERATOR_HOOK_MODES:
        verbose_proxy_logger.warning(
            "UnifiedLLMGuardrails: ignoring unknown iterator_hook_mode=%r "
            "(allowed: %s); falling back to %r.",
            mode,
            ", ".join(_VALID_ITERATOR_HOOK_MODES),
            ITERATOR_HOOK_MODE_MODERATION,
        )
        mode = ITERATOR_HOOK_MODE_MODERATION

    return mode


# Per-(choice, content) cursor of how many characters of the current
# guardrail-modified accumulated text have been emitted to the client.
# `content_idx` is `None` for the standard string-content case; for
# OpenAI multimodal list content it's the index of the text part. Reset
# at the start of each streaming hook invocation; not persisted across
# calls.
_TransformCursor = Dict[Tuple[int, Optional[int]], int]


def _iter_modified_chunk_content(
    responses_so_far: List[Any],
) -> List[Tuple[int, Optional[int], str]]:
    """Read the modified accumulated content from `responses_so_far`.

    `_apply_guardrail_responses_to_output_streaming` (and equivalents
    on other endpoint translations) puts the full guardrail-modified
    accumulated text in the FIRST chunk per choice and clears subsequent
    chunks to "". This helper extracts the modified text from chunk 0
    so transform-mode can compute the delta to yield.

    Returns a list of `(choice_idx, content_idx, modified_text)`
    tuples. `content_idx` is `None` for plain-string `delta.content` /
    `message.content`; an int for the multimodal list-content case.
    Choices that don't carry text (tool calls only, etc.) are skipped.

    Returns an empty list when `responses_so_far` has no chunks or the
    first chunk has no choices — the caller treats that as "nothing to
    yield this sample."
    """
    out: List[Tuple[int, Optional[int], str]] = []
    if not responses_so_far:
        return out
    first = responses_so_far[0]
    choices = getattr(first, "choices", None)
    if not choices:
        return out
    for choice_idx, choice in enumerate(choices):
        # Streaming chunks have `delta`; non-streaming have `message`.
        # We're called from the streaming path so prefer `delta` and
        # only fall back for safety.
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content is None:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None

        if isinstance(content, str):
            out.append((choice_idx, None, content))
        elif isinstance(content, list):
            for content_idx, content_item in enumerate(content):
                if isinstance(content_item, dict) and isinstance(
                    content_item.get("text"), str
                ):
                    out.append((choice_idx, content_idx, content_item["text"]))
    return out


def _build_transform_chunk(
    template: Any,
    choice_idx: int,
    content_idx: Optional[int],
    delta_text: str,
) -> Any:
    """Build a streaming chunk carrying just the delta we want to emit.

    `template` is an upstream chunk we deep-copy to inherit metadata
    (`id`, `model`, `created`, etc.). We then overwrite the chosen
    choice's content with `delta_text` and clear/replace any other
    text-bearing fields so the client only sees what we mean to send.

    Note this intentionally yields a single chunk per call; the caller
    yields one chunk per `(choice_idx, content_idx)` for which the
    delta is non-empty.
    """
    chunk = copy.deepcopy(template)
    choices = getattr(chunk, "choices", None) or []

    for emit_idx, choice in enumerate(choices):
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if emit_idx != choice_idx:
            # Suppress unrelated choices on this delta-only chunk.
            if hasattr(delta, "content"):
                delta.content = None
            continue

        if content_idx is None:
            delta.content = delta_text
        else:
            existing = getattr(delta, "content", None)
            if isinstance(existing, list) and 0 <= content_idx < len(existing):
                # Multimodal: replace just the targeted text part; clear
                # any other text parts on the same choice.
                for item_idx, content_item in enumerate(existing):
                    if not isinstance(content_item, dict):
                        continue
                    if "text" not in content_item:
                        continue
                    content_item["text"] = delta_text if item_idx == content_idx else ""
            else:
                # Couldn't find the targeted slot — fall back to
                # plain-string assignment so we still emit something
                # rather than dropping the delta silently.
                delta.content = delta_text

    return chunk


def _emit_transform_deltas(
    responses_so_far: List[Any],
    template: Any,
    cursor: _TransformCursor,
) -> List[Any]:
    """Compute and return the chunks to yield this sample for transform
    mode.

    Reads modified accumulated text from `responses_so_far[0]`, diffs
    against `cursor[(choice_idx, content_idx)]`, and emits one
    delta-only chunk per (choice, content) that has new content. Mutates
    `cursor` in place to advance the per-choice yielded-character count.

    Fail-open behavior:
      * If the modified text is shorter than what's already been yielded
        (a guardrail removed earlier characters) we can't un-yield, so
        we skip emission for that (choice, content) without advancing
        the cursor. The next sample's modified text will be diffed
        against the same cursor — if it grows back past the old
        position, the new tail is emitted.
      * Empty deltas are not yielded.
    """
    chunks: List[Any] = []
    for choice_idx, content_idx, modified in _iter_modified_chunk_content(
        responses_so_far
    ):
        key = (choice_idx, content_idx)
        already = cursor.get(key, 0)
        if len(modified) < already:
            # Fail-open: shrink-modified content has no clean delta.
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails transform mode: modified content "
                "shorter than already-yielded prefix for "
                "(choice=%s, content=%s); skipping emission this sample.",
                choice_idx,
                content_idx,
            )
            continue
        new_delta = modified[already:]
        if not new_delta:
            continue
        chunks.append(
            _build_transform_chunk(template, choice_idx, content_idx, new_delta)
        )
        cursor[key] = len(modified)
    return chunks


def _ensure_litellm_metadata(data: dict, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Populate data['litellm_metadata'] from user_api_key_dict if absent."""
    if "litellm_metadata" not in data:
        from litellm.llms.base_llm.guardrail_translation.base_translation import (
            BaseTranslation,
        )

        user_metadata = BaseTranslation.transform_user_api_key_dict_to_metadata(
            user_api_key_dict
        )
        if user_metadata:
            data["litellm_metadata"] = user_metadata


class UnifiedLLMGuardrails(CustomLogger):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "UnifiedLLMGuardrails initialized with optional_params: %s",
            self.optional_params,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        global endpoint_guardrail_translation_mappings
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("Running UnifiedLLMGuardrails pre-call hook")

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)
        if guardrail_to_apply is None:
            return data

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.pre_mcp_call

        if (
            guardrail_to_apply.should_run_guardrail(data=data, event_type=event_type)
            is not True
        ):
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Pre-call scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            return data

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )

        try:
            if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
                return data
        except ValueError:
            return data  # handle unmapped call types

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        _ensure_litellm_metadata(data, user_api_key_dict)

        data = await endpoint_translation.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=data.get("litellm_logging_obj"),
        )

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=guardrail_to_apply.guardrail_name
        )
        return data

    async def async_moderation_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> Any:
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """
        global endpoint_guardrail_translation_mappings

        verbose_proxy_logger.debug("Running UnifiedLLMGuardrails moderation hook")

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)
        if guardrail_to_apply is None:
            return data

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.during_mcp_call

        if (
            guardrail_to_apply.should_run_guardrail(data=data, event_type=event_type)
            is not True
        ):
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Pre-call scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            return data

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )
        if (
            call_type is not None
            and CallTypes(call_type) not in endpoint_guardrail_translation_mappings
        ):
            return data

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        _ensure_litellm_metadata(data, user_api_key_dict)

        return await endpoint_translation.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=data.get("litellm_logging_obj"),
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ) -> Any:
        """
        Runs on response from LLM API call

        It can be used to reject a response

        Uses Enkrypt AI guardrails to check the response for policy violations, PII, and injection attacks
        """
        global endpoint_guardrail_translation_mappings
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)

        if guardrail_to_apply is None:
            return

        if (
            guardrail_to_apply.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return

        verbose_proxy_logger.debug(
            "async_post_call_success_hook response: %s", response
        )

        call_type: Optional[CallTypesLiteral] = None
        if user_api_key_dict.request_route is not None:
            call_types = get_call_types_for_route(user_api_key_dict.request_route)
            if call_types is not None and len(call_types) > 0:  # type: ignore
                call_type = call_types[0]  # type: ignore
        if call_type is None:
            call_type = _infer_call_type(call_type=None, completion_response=response)  # type: ignore

        # Fallback: resolve call_type from logging_obj for pass-through endpoints
        if call_type is None:
            litellm_logging_obj = data.get("litellm_logging_obj")
            if (
                litellm_logging_obj is not None
                and getattr(litellm_logging_obj, "call_type", None)
                == CallTypes.pass_through.value
            ):
                call_type = CallTypes.pass_through.value

        if call_type is None:
            return response

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )

        if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
            return response

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        response = await endpoint_translation.process_output_response(
            response=response,  # type: ignore
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=data.get("litellm_logging_obj"),
            user_api_key_dict=user_api_key_dict,
            request_data=data,
        )
        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=guardrail_to_apply.guardrail_name
        )

        return response

    async def async_post_call_streaming_iterator_hook(  # noqa: PLR0915
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        See Aim guardrail implementation for an example - https://github.com/BerriAI/litellm/blob/d0e022cfacb8e9ebc5409bb652059b6fd97b45c0/litellm/proxy/guardrails/guardrail_hooks/aim.py#L168

        Triggered by mode: 'post_call'

        Supports sampling_rate parameter to control how often chunks are processed.
        sampling_rate=1 means every chunk, sampling_rate=5 means every 5th chunk, etc.
        """

        global endpoint_guardrail_translation_mappings

        guardrail_to_apply: CustomGuardrail = request_data.pop(
            "guardrail_to_apply", None
        )

        # Get streaming configuration from guardrail or optional_params
        sampling_rate = 5
        end_of_stream_only = False  # If True, only apply guardrail at end of stream

        if guardrail_to_apply is not None:
            # Check direct attributes on guardrail first
            sampling_rate = getattr(
                guardrail_to_apply, "streaming_sampling_rate", sampling_rate
            )
            end_of_stream_only = getattr(
                guardrail_to_apply, "streaming_end_of_stream_only", end_of_stream_only
            )

            # Also check guardrail_config dict if present
            guardrail_config = getattr(guardrail_to_apply, "guardrail_config", {})
            if isinstance(guardrail_config, dict):
                sampling_rate = guardrail_config.get(
                    "streaming_sampling_rate", sampling_rate
                )
                end_of_stream_only = guardrail_config.get(
                    "streaming_end_of_stream_only", end_of_stream_only
                )

        # Also check optional_params as fallback
        sampling_rate = self.optional_params.get(
            "streaming_sampling_rate", sampling_rate
        )
        end_of_stream_only = self.optional_params.get(
            "streaming_end_of_stream_only", end_of_stream_only
        )

        # Iterator-hook mode resolution. `transform` opts a guardrail
        # into yielding modified texts to the client (rather than the
        # historical moderation-only behavior of yielding originals).
        # See `_resolve_iterator_hook_mode` for the lookup chain.
        iterator_hook_mode = _resolve_iterator_hook_mode(
            guardrail_to_apply, self.optional_params
        )
        # Per-(choice, content_idx) yielded-character cursor used in
        # transform mode. See `_emit_transform_deltas`.
        transform_cursor: _TransformCursor = {}

        if guardrail_to_apply is None:
            async for item in response:
                yield item
            return

        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if (
            guardrail_to_apply.should_run_guardrail(
                data=request_data, event_type=event_type
            )
            is not True
        ):
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Post-call streaming scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            async for item in response:
                yield item
            return

        # Initialize translation mappings if needed
        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )

        # Infer call type from first chunk
        call_type = None
        chunk_counter = 0
        responses_so_far: List[Any] = []

        async for item in response:
            chunk_counter += 1
            responses_so_far.append(item)

            # Infer call type from first chunk if not already done
            if call_type is None and user_api_key_dict.request_route is not None:
                call_types = get_call_types_for_route(user_api_key_dict.request_route)
                if call_types is not None:
                    call_type = call_types[0].value

            if call_type is None:
                call_type = _infer_call_type(call_type=None, completion_response=item)  # type: ignore

            # If call type not supported, just pass through all chunks
            if (
                call_type is None
                or CallTypes(call_type) not in endpoint_guardrail_translation_mappings
            ):
                yield item
                async for remaining_item in response:
                    yield remaining_item
                return

            # If end_of_stream_only mode, yield chunks without processing.
            # In transform mode, end_of_stream_only buffers everything
            # until the post-loop final-flush block below — yielding
            # originals here would defeat the purpose.
            if end_of_stream_only:
                if iterator_hook_mode == ITERATOR_HOOK_MODE_MODERATION:
                    yield item
                continue

            # Process chunk based on sampling rate
            if chunk_counter % sampling_rate == 0:
                verbose_proxy_logger.debug(
                    "Processing streaming chunk %s (sampling_rate=%s) with guardrail %s",
                    chunk_counter,
                    sampling_rate,
                    guardrail_to_apply.guardrail_name,
                )

                # Deep-copy the current chunk before guardrail processing.
                # process_output_streaming_response modifies responses_so_far
                # in-place: it puts the combined guardrailed text in the first
                # chunk and clears all subsequent chunks to "". Without this
                # copy, yielding processed_items[-1] would yield an empty
                # string, permanently losing this chunk's content.
                original_item = copy.deepcopy(item)

                endpoint_translation = endpoint_guardrail_translation_mappings[
                    CallTypes(call_type)
                ]()

                try:
                    await endpoint_translation.process_output_streaming_response(
                        responses_so_far=responses_so_far,
                        guardrail_to_apply=guardrail_to_apply,
                        litellm_logging_obj=request_data.get("litellm_logging_obj"),
                        user_api_key_dict=user_api_key_dict,
                        request_data=request_data,
                    )
                except HTTPException as e:
                    # Response already started (we already yielded chunks); cannot send 400.
                    # For A2A (NDJSON), yield an in-stream JSON-RPC error so the client sees it.
                    if call_type is not None and CallTypes(call_type) in A2A_CALL_TYPES:
                        request_id = _get_a2a_request_id(responses_so_far, request_data)
                        detail = (
                            e.detail
                            if isinstance(e.detail, dict)
                            else {"message": str(e.detail)}
                        )
                        error_chunk = (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": request_id,
                                    "error": {
                                        "code": -32603,
                                        "message": detail.get(
                                            "error",
                                            detail.get("message", str(e.detail)),
                                        ),
                                        "data": {
                                            k: v
                                            for k, v in detail.items()
                                            if k not in ("error", "message")
                                        },
                                    },
                                }
                            )
                            + "\n"
                        )
                        yield error_chunk
                        return
                    raise

                if iterator_hook_mode == ITERATOR_HOOK_MODE_TRANSFORM:
                    # Yield the new tail of the guardrail-modified
                    # accumulated text (per choice / content index)
                    # rather than the original chunk. Buffered chunks
                    # since the last sample are drained here too — the
                    # cursor advances by exactly the number of new
                    # characters, regardless of how many upstream
                    # chunks contributed them.
                    for delta_chunk in _emit_transform_deltas(
                        responses_so_far, original_item, transform_cursor
                    ):
                        yield delta_chunk
                else:
                    yield original_item
            else:
                # Between samples: moderation mode forwards the
                # original chunk immediately; transform mode buffers
                # (the chunk was already appended to `responses_so_far`
                # above and will be drained at the next sample point or
                # at end-of-stream).
                if iterator_hook_mode == ITERATOR_HOOK_MODE_MODERATION:
                    yield item

        # Stream has ended - do final processing with all collected chunks
        if (
            call_type is not None
            and CallTypes(call_type) in endpoint_guardrail_translation_mappings
        ):
            verbose_proxy_logger.debug(
                "Processing final streaming response with all %s chunks for guardrail %s",
                len(responses_so_far),
                guardrail_to_apply.guardrail_name,
            )

            endpoint_translation = endpoint_guardrail_translation_mappings[
                CallTypes(call_type)
            ]()

            try:
                await endpoint_translation.process_output_streaming_response(
                    responses_so_far=responses_so_far,
                    guardrail_to_apply=guardrail_to_apply,
                    litellm_logging_obj=request_data.get("litellm_logging_obj"),
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                )
            except HTTPException as e:
                if call_type is not None and CallTypes(call_type) in A2A_CALL_TYPES:
                    request_id = _get_a2a_request_id(responses_so_far, request_data)
                    detail = (
                        e.detail
                        if isinstance(e.detail, dict)
                        else {"message": str(e.detail)}
                    )
                    error_chunk = (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": detail.get(
                                        "error", detail.get("message", str(e.detail))
                                    ),
                                    "data": {
                                        k: v
                                        for k, v in detail.items()
                                        if k not in ("error", "message")
                                    },
                                },
                            }
                        )
                        + "\n"
                    )
                    yield error_chunk
                    return
                else:
                    raise

            if iterator_hook_mode == ITERATOR_HOOK_MODE_TRANSFORM and responses_so_far:
                # Final transform-mode flush. Drain any remaining new
                # tail of the guardrail-modified accumulated text, then
                # forward a content-cleared copy of the last upstream
                # chunk so `finish_reason` / `usage` / other terminal
                # metadata reaches the client. The terminal chunk is
                # always emitted, even if no fresh delta content was
                # produced this sample (otherwise SSE clients would
                # never see the stream terminate cleanly under transform
                # mode for guardrails that don't add content).
                template = responses_so_far[-1]
                for delta_chunk in _emit_transform_deltas(
                    responses_so_far, template, transform_cursor
                ):
                    yield delta_chunk

                terminal = copy.deepcopy(template)
                for choice in getattr(terminal, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    # Wipe content but preserve finish_reason / usage /
                    # role / tool_calls / anything else attached to the
                    # last upstream chunk.
                    if hasattr(delta, "content"):
                        delta.content = None
                yield terminal
