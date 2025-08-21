# Deepgram Integration for ASR Benchmarking

This document describes the Deepgram integration added to the ASR benchmarking script.

## Setup

1. **Install the Deepgram SDK:**
   ```bash
   pip install deepgram-sdk
   # or install all API requirements:
   pip install -r ../requirements/requirements-api.txt
   ```

2. **Set your Deepgram API key:**
   ```bash
   export DEEPGRAM_API_KEY="your_actual_api_key_here"
   ```
   
   You can obtain an API key by signing up at https://deepgram.com

## Supported Models

The integration supports all Deepgram models. Use the format `deepgram/<model_name>` or `deepgram/<model_name>:<tier>`:

- `deepgram/nova-2` - Latest Nova 2 model (recommended)
- `deepgram/nova-2:enhanced` - Nova 2 with enhanced tier for better accuracy
- `deepgram/nova` - Previous Nova version
- `deepgram/whisper-large` - Deepgram's Whisper Large model
- `deepgram/whisper-medium` - Deepgram's Whisper Medium model
- `deepgram/whisper-small` - Deepgram's Whisper Small model  
- `deepgram/whisper-base` - Deepgram's Whisper Base model
- `deepgram/whisper-tiny` - Deepgram's Whisper Tiny model
- `deepgram/base` - Base model

### Tier Options

You can specify a tier by appending `:<tier>` to the model name:
- `:nova` - Standard tier (default)
- `:enhanced` - Enhanced tier for better accuracy

Example: `deepgram/nova-2:enhanced`

## Usage

### Running a benchmark with Deepgram:

```bash
python run_eval.py \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.clean" \
    --model_name="deepgram/nova-2" \
    --max_workers=32
```

### Using URL mode (faster for remote datasets):

```bash
python run_eval.py \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.clean" \
    --model_name="deepgram/nova-2" \
    --use_url \
    --max_workers=32
```

### Running multiple Deepgram models:

Edit `run_api.sh` and uncomment the Deepgram models you want to test:

```bash
MODEL_IDs=(
    "deepgram/nova-2"
    "deepgram/nova-2:enhanced"
    "deepgram/nova"
    "deepgram/whisper-large"
    # ... other models
)
```

Then run:
```bash
bash run_api.sh
```

## Testing the Integration

A test script is provided to verify the Deepgram integration:

```bash
python test_deepgram.py
```

This will test:
- API key configuration
- Local file transcription (if audio libraries are installed)
- URL-based transcription
- Multiple Deepgram models

## Features

- **Automatic retry logic**: Handles transient API errors
- **Caching support**: Results are cached to avoid redundant API calls
- **Both file and URL modes**: Supports local files and remote URLs
- **Multiple models**: Test different Deepgram models in the same run
- **Tier selection**: Choose between standard and enhanced tiers
- **Language support**: Currently configured for English (en)
- **Smart formatting**: Includes punctuation and formatting

## Troubleshooting

1. **API Key Issues:**
   - Ensure `DEEPGRAM_API_KEY` is set correctly
   - Check that your API key has sufficient credits

2. **Import Errors:**
   - Install the SDK: `pip install deepgram-sdk`
   
3. **Transcription Errors:**
   - Check your internet connection
   - Verify the audio file format is supported
   - Check Deepgram service status

## API Rate Limits

Deepgram has generous rate limits, but for large-scale benchmarking:
- Consider using `--max_workers` to control concurrency
- Monitor your API usage in the Deepgram console
- Use caching to avoid redundant API calls

## Cost Considerations

- Deepgram charges per minute of audio processed
- Enhanced tier costs more but provides better accuracy
- Check current pricing at https://deepgram.com/pricing
- Use the caching feature to minimize costs during development
