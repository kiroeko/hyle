"""Portable Qwen3.5 image preprocessing and ordered language prompt expansion."""
from __future__ import annotations

from io import BytesIO
import math

from ...text.contracts import ImageInput, PreparedPrompt, PromptAttachment
from .vision_preparation import read_recipe


class ImageProcessor:
    def __init__(self, model, tokenizer, *, context):
        config, self.vision, self.processor = read_recipe(model)
        self.tokenizer, self.context = tokenizer, context
        self.image_id, self.start_id, self.end_id = (config[key] for key in
            ('image_token_id', 'vision_start_token_id', 'vision_end_token_id'))

    def patches(self, image: ImageInput, *, max_rows=None):
        import numpy as np
        from PIL import Image, ImageOps, UnidentifiedImageError
        try:
            with Image.open(BytesIO(image.data)) as original:
                if Image.MIME.get(original.format) != image.media_type:
                    raise ValueError('image bytes do not match their declared media type')
                if getattr(original, 'n_frames', 1) != 1:
                    raise ValueError('animated images require a video input contract')
                if original.width * original.height > 32 * 1024 * 1024:
                    raise ValueError('decoded image pixel limit exceeded')
                rgb = ImageOps.exif_transpose(original).convert('RGB')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError('image could not be decoded') from error
        p, m, t = (self.vision[key] for key in ('patch_size', 'spatial_merge_size', 'temporal_patch_size'))
        width, height = rgb.size
        factor = p * m
        if max(width, height) / min(width, height) > 200: raise ValueError('image aspect ratio exceeds the processor contract')
        maximum = min(self.processor['size']['longest_edge'], (self.context if max_rows is None else max_rows) * factor * factor)
        if image.detail=='low': maximum=min(maximum,self.processor['size']['shortest_edge'])
        minimum = min(self.processor['size']['shortest_edge'], maximum)
        rh, rw = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
        if rh * rw > maximum:
            beta = math.sqrt(height * width / maximum)
            rh, rw = max(factor, math.floor(height / beta / factor)*factor), max(factor, math.floor(width / beta / factor)*factor)
        elif rh * rw < minimum:
            beta = math.sqrt(minimum / (height * width))
            rh, rw = math.ceil(height * beta / factor)*factor, math.ceil(width * beta / factor)*factor
        if rh*rw>maximum: raise ValueError('image aspect ratio cannot fit the remaining prompt budget')
        rgb = rgb.resize((rw, rh), Image.Resampling.BICUBIC)
        # Match the source's separate rescale, normalization and temporal repeat.
        pixels = np.asarray(rgb).astype(np.float32) * np.float32(1/255)
        pixels = (pixels - np.float32(.5)) / np.float32(.5)
        frames = np.repeat(pixels.transpose(2, 0, 1)[None], t, axis=0)
        gh, gw = rh // p, rw // p
        patches = frames.reshape(1, t, 3, gh//m, m, p, gw//m, m, p)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh*gw, 3*t*p*p)
        return (1, gh, gw), patches.astype('<f4', copy=False).tobytes()

    def prepare(self, messages, *, template=None, continue_final_message=False, max_tokens=None,
                template_options=None, generation_prefix=''):
        images = []
        for message in messages:
            if isinstance(message['content'], list):
                images.extend(part['image'] for part in message['content'] if part['type'] == 'image')
        if template_options is not None:
            rendered = self.tokenizer.apply_chat_template(list(messages), chat_template=template,
                tokenize=False, add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message, **template_options)
            tokenized = self.tokenizer.encode(rendered + generation_prefix, add_special_tokens=False)
        else:
            tokenized = self.tokenizer.apply_chat_template(list(messages), chat_template=template,
                tokenize=True, add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message)
        if hasattr(tokenized, 'input_ids'): tokenized = tokenized.input_ids
        budget=(self.context if max_tokens is None else max_tokens)-(len(tokenized)-len(images))
        if budget<len(images): raise ValueError('text and output leave no image context capacity')
        ids, attachments, index = [], [], 0
        for at, token in enumerate(tokenized):
            if token != self.image_id:
                ids.append(token); continue
            if (index >= len(images) or at == 0 or at + 1 == len(tokenized) or
                    tokenized[at-1] != self.start_id or tokenized[at+1] != self.end_id):
                raise ValueError('template image placeholders differ from the ordered image inputs')
            grid, payload = self.patches(images[index],max_rows=budget//(len(images)-index)); index += 1
            rows = grid[1] * grid[2] // self.vision['spatial_merge_size'] ** 2
            budget-=rows
            attachments.append(PromptAttachment('qwen35.image.patches.f32.v1', len(ids), rows, grid, payload))
            ids.extend([self.image_id] * rows)
        if index != len(images): raise ValueError('template did not preserve every image input')
        return PreparedPrompt(ids, tuple(attachments))
