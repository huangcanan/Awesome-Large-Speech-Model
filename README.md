# Awesome-Large-Speech-Model
A repository used to organize content related to Large Speech(Audio) Model, including paper, data, applications, tools and so on.

This repo will be updated continuously ...

## outline
- [Papers](#paper)
  - [Survey](#paper1)
  - [Speech/Audio-Extended LLM](#paper2)
  - [Speech Recognition & Translation](#paper3)
  - [Speech Synthesis](#paper4)
  - [Self-supervised Pretrain Speech Model](#paper5)
  - [Speech Tokenization/Discretization](#paper6)
  - [Audio/Music/Others](#paper7)
- [Open-source Large Speech/Audio Models](#model)
- [Data](#data)
- [Tools](#tool)
- [Applications](#application)

## Papers<a id="paper"></a>
### Survey<a id="paper1"></a>
- **Recent Advances in Speech Language Models: A Survey** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2410.03751)]
- **A Comprehensive Survey on Deep Multimodal Learning with Missing Modality** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.07825)]
- **A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.01319)]
- **MM-LLMs: Recent Advances in MultiModal Large Language Models** , 2024 , ACL [[paper](https://arxiv.org/pdf/2401.13601)] [[github.io](https://mm-llms.github.io/)]
- **A Survey on Multimodal Large Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.13549)] [[github](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)]
- **Sparks of Large Audio Models: A Survey and Outlook** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.12792)]

### Speech/Audio-Extended Large Language Model<a id="paper2"></a>
- **EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.18042)] [[github.io](https://emova-ollm.github.io/)]
- **Moshi: a speech-text foundation model for real-time dialogue** , 2024 [[paper](https://kyutai.org/Moshi.pdf)] [[github](https://github.com/kyutai-labs/moshi)]
- **LLaMA-Omni: Seamless Speech Interaction with Large Language Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.06666)] [[github](https://github.com/ictnlp/LLaMA-Omni)]
- **MooER: LLM-based Speech Recognition and Translation Models from Moore Threads** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.05101)]  [[github](https://github.com/MooreThreads/MooER)]
- **Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.16725)] [[github](https://github.com/gpt-omni/mini-omni)]
- **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context** , 2024 , arXiv [[paper](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)] 
- **The Llama 3 Herd of Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.21783)]
- **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2405.05945)] [[github](https://github.com/Alpha-VLLM/Lumina-T2X)]
- **VITA: Towards Open-Source Interactive Omni Multimodal LLM** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.05211)] [[github.io](https://vita-home.github.io/)]
- **Qwen2-Audio Technical Report** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.10759)] [[github](https://github.com/QwenLM/Qwen2-Audio)]
- **NExT-GPT: Any-to-Any Multimodal LLM** , 2024 , ICML [[paper](https://arxiv.org/pdf/2309.05519)] [[github](https://github.com/NExT-GPT/NExT-GPT)]
- **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** , 2024 , arXiv [[paper](https://arxiv.org/abs/2402.12226)] [[github](https://github.com/OpenMOSS/AnyGPT)]
- **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2402.08846)] [[github](https://github.com/X-LANCE/SLAM-LLM)]
- **SALMONN: Towards Generic Hearing Abilities for Large Language Models** , 2024 , ICLR [[paper](https://openreview.net/pdf?id=14rn7HpKVk)] [[github](https://github.com/bytedance/SALMONN)]
- **Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.09093)] [[github](https://github.com/lyuchenyang/Macaw-LLM)]
- **LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2310.04673)] [[github.io](https://lauragpt.github.io/)] 
- **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2311.07919)] [[github](https://github.com/QwenLM/Qwen-Audio)]
- **SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities** , 2023 , EMNLP [[paper](https://arxiv.org/pdf/2305.11000)] [[github](https://github.com/0nutation/SpeechGPT)]
- **Pengi: An Audio Language Model for Audio Tasks** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2305.11834)] [[github](https://github.com/microsoft/Pengi)]
- **AudioPaLM: A Large Language Model That Can Speak and Listen** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.12925)] [[github.io](https://google-research.github.io/seanet/audiopalm/examples/)]

### Speech Recognition & Translation<a id="paper3"></a>
- **Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.04675)] [[github.io](https://bytedancespeech.github.io/seedasr_tech_report/)]
- **OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2401.16658)] [[github](https://github.com/espnet/espnet/tree/master/egs2/owsm_v3.1/s2t1)]
- **Seamless: Multilingual Expressive and Streaming Speech Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2312.05187)] [[github](https://github.com/facebookresearch/seamless_communication)] , Remark: SeamlessM4T v2
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.11596)] [[github](https://github.com/facebookresearch/seamless_communication)]
- **Robust Speech Recognition via Large-Scale Weak Supervision** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.04356)] [[github](https://github.com/openai/whisper)] , Remark: Whisper

### Speech Synthesis<a id="paper4"></a>
- **Natural language guidance of high-fidelity text-to-speech with synthetic annotations** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2402.01912)]  [[demo](https://www.text-description-to-speech.com/)]
- **CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.05407)] [[github](https://github.com/FunAudioLLM/CosyVoice)] [[github.io](https://fun-audio-llm.github.io/)]
- **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.02430)] [[github.io](https://bytedancespeech.github.io/seedtts_tech_report/)]
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2403.03100)] [[github](https://speechresearch.github.io/naturalspeech3/)]
- **Boosting Large Language Model for Speech Synthesis: An Empirical Study** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2401.00246)]
- **SoundStorm: Efficient Parallel Audio Generation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2305.09636)] [[github.io](https://google-research.github.io/seanet/soundstorm/examples/)]
- **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.02111)] [[github](https://github.com/lifeiteng/vall-e)] , Remark: VALL-E
- **PromptTTS: Controllable Text-to-Speech with Text Descriptions** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2211.12171)] [[github.io](https://speechresearch.github.io/prompttts/)]

### Self-supervised Pretrain Speech Model<a id="paper5"></a>
- **FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.03283)] [[github](https://github.com/FireRedTeam/FireRedTTS)] [[github.io](https://fireredteam.github.io/demos/firered_tts/)]
- **BEATs: Audio Pre-Training with Acoustic Tokenizers** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.09058)] [[github](https://github.com/microsoft/unilm/tree/master/beats)]
- **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2202.03555)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
- **W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2108.06209)] 
- **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** , 2021 , arXiv [[paper](https://arxiv.org/abs/2110.13900)]
- **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2106.07447)]
- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** , 2020 , NeurIPS [[paper](https://arxiv.org/pdf/2006.11477)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]
- **wav2vec: Unsupervised Pre-Training for Speech Recognition** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1904.05862)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]

### Speech Tokenization/Discretization<a id="paper6"></a>
- **WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.16532)] [[github](https://github.com/jishengpeng/WavTokenizer)] [[github.io](https://wavtokenizer.github.io/)]
- **RepCodec: A Speech Representation Codec for Speech Tokenization** , 2024 , ACL [[paper](https://arxiv.org/pdf/2309.00169)] [[github](https://github.com/mct10/RepCodec)]
- **FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2309.07405)] [[github](https://github.com/modelscope/FunCodec)] [[github.io](https://funcodec.github.io/)]
- **Single-Codec: Single-Codebook Speech Codec towards High-Performance Speech Generation** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2406.07422)] [[github.io](https://kkksuper.github.io/Single-Codec/)]
- **ToneUnit: A Speech Discretization Approach for Tonal Language Speech Synthesis** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.08989)] [[github.io](https://toneunit1225.github.io/)]
- **SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models** , 2024 , ICLR [[paper](https://arxiv.org/pdf/2308.16692)] [[github](https://github.com/ZhangXInFD/SpeechTokenizer)]
- **Finite Scalar Quantization: VQ-VAE Made Simple** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2309.15505)] [[github](https://github.com/google-research/google-research/tree/master/fsq)]
- **High-Fidelity Audio Compression with Improved RVQGAN** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2306.06546)] [[github](https://github.com/descriptinc/descript-audio-codec)]
- **High Fidelity Neural Audio Compression** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2210.13438)] [[github](https://github.com/facebookresearch/encodec)] , Remark: Encodec
- **vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1910.05453)]
- **Neural Discrete Representation Learning** , 2017 , NeurIPS, [[paper](https://arxiv.org/pdf/1711.00937)] [[code](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)] , Remark: VQ-VAE

### Audio/Music/Others<a id="paper7"></a>
- **Seed-Music: A Unified Framework for High Quality and Controlled Music Generation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.09214)]
- **QA-MDT: Quality-aware Masked Diffusion Transformer for Enhanced Music Generation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2405.15863)] [[github](https://github.com/ivcylc/qa-mdt)] [[github](https://qa-mdt.github.io/)]
- **M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2311.11255)] [[github](https://github.com/shansongliu/M2UGen)]
- **SingSong: Generating musical accompaniments from singing** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.12662)] [[example](https://storage.googleapis.com/sing-song/index.html)]
- **Simple and Controllable Music Generation** , 2023 , NeurIPS , [[paper](https://arxiv.org/pdf/2306.05284)] [[github](https://github.com/facebookresearch/audiocraft)]
- **AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models** , 2023 , NeurIPS , [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/e1b619a9e241606a23eb21767f16cf81-Paper-Conference.pdf)] [[github.io](https://audioldm.github.io/)]
- **Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2304.13731)] [[github](https://github.com/declare-lab/tango)] [[github.io](https://tango-web.github.io/)]
- **MusicLM: Generating Music From Text** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.11325)] [[github.io](https://google-research.github.io/seanet/musiclm/examples/)]
- **MuLan: A Joint Embedding of Music Audio and Natural Language** , 2022 , ISMIR [[paper](https://arxiv.org/pdf/2208.12415)] [[github](https://github.com/lucidrains/musiclm-pytorch)]
- **AudioLM: a Language Modeling Approach to Audio Generation** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2209.03143)] [[github.io](https://google-research.github.io/seanet/audiolm/examples/)]

## Open-source Large Speech/Audio Models<a id="model"></a>
- **Whisper** 
  - Developed by OpenAI [[github](https://github.com/openai/whisper)] ![](https://img.shields.io/github/stars/openai/whisper.svg)
  - Introduction: Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform **multilingual speech recognition**, **speech translation**, and **language identification**. It supports speech recognition for 100 languages. Five model sizes are trained, including tiny(39 M), base(74 M), small(244 M), medium(769 M), large(1550 M) and the newest large v3 model is trained on 5M hours of labeled data.
- **Seamless**
  - Developed by Meta [[github](https://github.com/facebookresearch/seamless_communication)] ![](https://img.shields.io/github/stars/facebookresearch/seamless_communication.svg)
  - Introduction: Seamless is a family of AI models that enable more natural and authentic communication across languages. It contains three main models, SeamlessM4T, SeamlessExpressive and SeamlessStreaming. **SeamlessM4T** is a massive multilingual multimodal machine translation model supporting around 100 languages. SeamlessM4T serves as foundation for **SeamlessExpressive**, a model that preserves elements of prosody and voice style across languages and **SeamlessStreaming**, a model supporting simultaneous translation and streaming ASR for around 100 languages. SeamlessExpressive and SeamlessStreaming are combined into Seamless, a unified model featuring multilinguality, real-time and expressive translations.
- **SenseVoice**
  - Developed by Alibaba [[github](https://github.com/FunAudioLLM/SenseVoice)] ![](https://img.shields.io/github/stars/FunAudioLLM/SenseVoice.svg)
  - Introduction: SenseVoice is a speech foundation model with multiple speech understanding capabilities, including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED).
- **CosyVoice**
  - Developed by Alibaba [[github](https://github.com/FunAudioLLM/CosyVoice)] ![](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice.svg)
- **Qwen2-Audio**
  - Developed by Alibaba [[github](https://github.com/QwenLM/Qwen2-Audio)] ![](https://img.shields.io/github/stars/QwenLM/Qwen2-Audio.svg)
  - Introduction: The latest progress of Qwen-Audio, a large-scale audio-language model called Qwen2-Audio, which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. Two distinct audio interaction modes are introductioned: (1) voice chat: users can freely engage in voice interactions with Qwen2-Audio without text input; (2) audio analysis: users could provide audio and text instructions for analysis during the interaction; Two models are released of the Qwen2-Audio series: Qwen2-Audio-7B and Qwen2-Audio-7B-Instruct.
- **SALMONN**
  - Developed by Tsinghua University and ByteDance [[github](https://github.com/bytedance/SALMONN)] ![](https://img.shields.io/github/stars/bytedance/SALMONN.svg)
  - Introduction: SALMONN is a large language model (LLM) enabling speech, audio events, and music inputs. Instead of speech-only input or audio-event-only input, SALMONN can perceive and understand all kinds of audio inputs and therefore obtain emerging capabilities such as multilingual speech recognition and translation and audio-speech co-reasoning.
- **Reverb**
  - Developed by Rev [[github](https://github.com/revdotcom/reverb)] ![](https://img.shields.io/github/stars/revdotcom/reverb.svg)
  - Introduction: Open source inference and evaluation code for Rev's state-of-the-art speech recognition and diarization models. The speech recognition (ASR) code uses the WeNet framework and the speech diarization code uses the Pyannote framework. Reverb ASR was trained on **200,000 hours** of English speech, all expertly **transcribed by humans**. The speech recognition models released outperform all existing open source speech recognition models across a variety of long-form speech recognition domains.
- **NVIDIA NeMo ASR**
  - Developed by NVIDIA [[github](https://github.com/NVIDIA/NeMo)] ![](https://img.shields.io/github/stars/NVIDIA/NeMo.svg)
  - Introduction: NVIDIA NeMo team released a number of inference optimizations for CTC, RNN-T, and TDT models that resulted in up to 10x inference speed-up. These models now exceed an inverse real-time factor (RTFx) of 2,000, with some reaching RTFx of even 6,000.

## Data<a id="data"></a>
Only common benchmarks and large-scale training datasets are listed here.
- Speech Recognition & Translation
- Speech Synthesis
- Audio Caption

|datasets|language|task|duration/hours|
| :---- | :---: | :---: | :---: |
| Librispeech [[download](https://www.openslr.org/12)] | English | ASR | 960 |
| Aishell-1 [[dowmload](https://www.openslr.org/33/)] | Mandarin | ASR | 170 |
| GigaSpeech [[download](https://github.com/SpeechColab/GigaSpeech)] | English | ASR | 10k |
| WenetSpeech [[download](https://github.com/wenet-e2e/WenetSpeech)] | Mandarin | ASR | 10k |
| Libri-Light [[download](https://github.com/facebookresearch/libri-light)] | English | ASR | 60K |
| Common Voice [[download](https://commonvoice.mozilla.org/zh-CN/datasets)] | Multilingual | ASR | >20k |
| MuST-C/MuST-C v2 | Multilingual | ST | 500 |
| CoVoST/CoVoST-2 [[download](https://github.com/facebookresearch/covost)] | Multilingual | ST | 2880 |
| GigaST [[download](https://github.com/bytedance/neurst/tree/master/datasets/GigaST)] | English-German/Chinese | ST | 10k |
| YODAS [[download](https://huggingface.co/datasets/espnet/yodas)] | Multilingual | ASR/Self-supervised | >500k |
| MSR-86K [[download](https://huggingface.co/datasets/Alex-Song/MSR-86K)] | Multilingual | ASR | 86k |
| LibriTTS [[download](https://www.openslr.org/60/)] | English | TTS | 585 |
| WenetSpeech4TTS [[download](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS)] | Mandarin | TTS | 10k |
| Emilia [[download](https://huggingface.co/datasets/amphion/Emilia-Dataset)] | Multilingual | TTS | 101k |
| AudioCaps [[download](https://huggingface.co/datasets/d0rj/audiocaps)] | - | AC | - |

## Tools<a id="tool"></a>
- **Silero VAD**: A pre-trained enterprise-grade Voice Activity Detector. [[github](https://github.com/snakers4/silero-vad)]
- **ESPnet**: End-to-end speech processing toolkit [[github](https://github.com/espnet/espnet)]
- **Amphion**: An Open-Source Audio, Music, and Speech Generation Toolkit [[github](https://github.com/open-mmlab/Amphion)]
- **Huggingface speech-to-speech**: An effort for an open-sourced and modular GPT-4o. [[github](https://github.com/huggingface/speech-to-speech)]
- **faster-whisper**: Faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models. [[github](https://github.com/SYSTRAN/faster-whisper)]
- **whisper_streaming**: Whisper realtime streaming for long speech-to-text transcription and translation. [[github](https://github.com/ufal/whisper_streaming)]
- **SLAM-LLM**: A deep learning toolkit that allows researchers and developers to train custom multimodal large language model (MLLM), focusing on Speech, Language, Audio, Music processing. [[github](https://github.com/X-LANCE/SLAM-LLM)]
- **swift**: Supports training(PreTraining/Fine-tuning/RLHF), inference, evaluation and deployment of 300+ LLMs and 50+ MLLMs (multimodal large models). [[github](https://github.com/modelscope/ms-swift)] [[DOCUMENTATION](https://swift.readthedocs.io/zh-cn/latest/index.html)]
- **Huggingface Parler-TTS**: A lightweight text-to-speech (TTS) model that can generate high-quality, natural sounding speech in the style of a given speaker (gender, pitch, speaking style, etc). [[github](https://github.com/huggingface/parler-tts)] [[demo](https://huggingface.co/spaces/parler-tts/parler_tts)]
- **RTC**: Real-time communication (RTC) is a real-time communication technology that allows the exchange of audio, video, or data between computers or other devices. You can find some demos in the [github](https://github.com/webrtc)  open-sourced by google. And many companies are also provide RTC PaaS survices.
- **edge-tts**: edge-tts is a Python module that allows you to use Microsoft Edge's online text-to-speech service from within your Python code or using the provided edge-tts or edge-playback command. [[github](https://github.com/rany2/edge-tts)]

## Applications<a id="application"></a>
- **GPT-4o** [[link](https://openai.com/index/hello-gpt-4o/)]
- **Moshi** [[link](https://moshi.chat/)]
- **heygen** [[link](https://www.heygen.com/)]
- **Hume AI** [[link](https://www.hume.ai/)]
- **Suno** [[link](https://suno.com/)]
- **Google Illuminate** [[link](https://illuminate.google.com/home?pli=1)]
- **ElevenLabs** [[link](https://elevenlabs.io/)]
