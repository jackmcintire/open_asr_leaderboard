import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import warnings

MODEL_ID = "aquavoice/sweet-lion-95"


class OptimizedWhisper:
    def __init__(
        self,
        model_id=MODEL_ID,
        device=None,
        dtype=torch.float16,
        chunk_length_s=30.0,  # Whisper window
        stride_s=0.0,  # set to 1-5s if you want overlap+merge
        batch_chunks=1,  # >1 for throughput if you want; keeps shapes static if constant
        language="en",
        task="transcribe",
    ):
        self.model_id = model_id
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.chunk_length_s = float(chunk_length_s)
        self.stride_s = float(stride_s)
        self.batch_chunks = int(batch_chunks)
        self.language = language
        self.task = task

        print(f"[Optimized] Loading model '{model_id}' on {self.device}...")

        self._apply_backend_optimizations()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.processor = AutoProcessor.from_pretrained(
                model_id, token=os.getenv("HF_TOKEN")
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                token=os.getenv("HF_TOKEN"),
            ).to(self.device)

        self.model.eval()

        # Prime decoder with language/task so we skip language detection and extra tokens.
        # try:
        #     forced = self.processor.get_decoder_prompt_ids(
        #         language=self.language, task=self.task
        #     )
        #     # prefer putting it on the generation config to keep generate() signature stable
        #     self.model.generation_config.forced_decoder_ids = forced
        # except Exception:
        #     pass  # some community models might not ship the helper; it's fine

        # Log size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[Optimized] Model loaded: {param_count / 1e9:.2f}B parameters")

        self._compile_model()
        self._warmup()

    def _apply_backend_optimizations(self):
        if self.device == "cuda":
            print("[Optimized] Applying backend optimizations...")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            try:
                if hasattr(torch.cuda, "memory"):
                    torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")
            except Exception:
                pass
            torch.cuda.empty_cache()
            print("[Optimized] Backend optimizations applied")

    def _compile_model(self):
        print("[Optimized] Compiling model with torch.compile...")
        try:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=True, dynamic=False
            )
            print("[Optimized] Model compilation successful")
        except Exception as e:
            print(f"[Optimized] Warning: torch.compile failed: {e}")
            print("[Optimized] Falling back to eager mode")

    def _warmup(self):
        print("[Optimized] Running warmup...")
        # Your model uses 128 mel bins and 30s windows -> 3000 frames (hop=160 @ 16 kHz)
        dummy_input = torch.randn(1, 128, 3000, dtype=self.dtype, device=self.device)
        with torch.inference_mode():
            _ = self.model.generate(dummy_input, max_new_tokens=1)
            _ = self.model.generate(dummy_input, max_new_tokens=1)
            _ = self.model.generate(dummy_input, max_new_tokens=1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print("[Optimized] Warmup complete")

    # ---------- NEW: chunker & simple merge ----------

    @staticmethod
    def _iter_audio_chunks(
        waveform: torch.Tensor, sr: int, chunk_length_s: float, stride_s: float
    ):
        """
        Yields contiguous chunks of raw audio (mono) as tensors in [samples], last chunk shorter is OK.
        """
        total = waveform.shape[-1]
        chunk_len = int(round(chunk_length_s * sr))
        stride = int(round(stride_s * sr))
        step = chunk_len - stride if stride > 0 else chunk_len

        start = 0
        while start < total:
            end = min(start + chunk_len, total)
            yield waveform[..., start:end]
            if end == total:
                break
            start += step

    @staticmethod
    def _merge_texts(pieces, max_overlap_chars=200):
        """
        Very light de-dup for overlapped chunks: trims the longest suffix/prefix match.
        If stride_s == 0, this just joins with spaces.
        """
        if not pieces:
            return ""
        merged = pieces[0].strip()
        for nxt in pieces[1:]:
            nxt = nxt.strip()
            # find the longest overlap between the end of merged and the start of nxt
            window = merged[-max_overlap_chars:]
            k = min(len(window), len(nxt))
            cut = 0
            for L in range(k, 0, -1):
                if window[-L:] == nxt[:L]:
                    cut = L
                    break
            merged = (merged + " " + nxt[cut:]).strip()
        return merged

    # ---------- UPDATED: transcribe over all chunks ----------

    def transcribe(self, audio_input, sr=None):
        import torchaudio

        # Handle both file paths and raw audio arrays
        if isinstance(audio_input, str):
            # File path - existing logic
            audio, sr = torchaudio.load(audio_input)
        else:
            # Raw audio array
            audio = (
                torch.tensor(audio_input)
                if not isinstance(audio_input, torch.Tensor)
                else audio_input
            )
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            sr = sr or 16000

        # Resample to 16k if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
            sr = 16000

        # To mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # List of decoded texts
        texts = []

        # (Optional) small, fixed-size batching for throughput while keeping shapes stable for compile
        batch = []
        for chunk in self._iter_audio_chunks(
            audio.squeeze(0), sr, self.chunk_length_s, self.stride_s
        ):
            batch.append(chunk)
            if len(batch) == self.batch_chunks:
                texts.extend(self._transcribe_batch(batch, sr))
                batch = []
        if batch:
            texts.extend(self._transcribe_batch(batch, sr))

        # Merge
        return self._merge_texts(
            texts, max_overlap_chars=200 if self.stride_s > 0 else 0
        )

    def _transcribe_batch(self, raw_chunks, sr):
        """
        raw_chunks: list[Tensor[samples]]
        Returns list[str] of decoded text per chunk.
        """
        # Convert tensors to numpy arrays for the processor
        numpy_chunks = [chunk.numpy() for chunk in raw_chunks]

        # The processor will pad/trim each chunk to exactly one 30s window (3000 frames).
        inputs = self.processor(numpy_chunks, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(
            self.device, dtype=self.dtype, non_blocking=True
        )

        gen_kwargs = dict(
            max_new_tokens=256,  # per 30s window; raise if your speech is very dense
            num_beams=1,
            do_sample=False,
            use_cache=True,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(input_features, **gen_kwargs)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
