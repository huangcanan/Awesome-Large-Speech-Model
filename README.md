# Awesome-Large-Speech-Model
A repository used to organize content related to Large Speech(Audio) Model, including paper, data, applications, tools and so on.

This repo will be updated continuously ...

## outline
- [Papers](#paper)
  - [Survey](#paper1)
  - [Speech/Audio-Extended LLM](#paper2)
  - [Speech Recognition & Translation](#paper3)
  - [Speech Synthesis](#paper4)
  - [Self-supervised Pre-trained Speech Model](#paper5)
  - [Speech Tokenization/Discretization](#paper6)
  - [Music Model](#paper7)
  - [Speech/Audio/Music Data](#paper8)
  - [Benchmark](#paper9)
  - [Safety & Risk](#paper10)
  - [Others](#paper11)
- [Open-source Large Speech/Audio Models](#model)
- [Datasets](#datasets)
- [Tools](#tool)
- [Applications](#application)

## Papers<a id="paper"></a>
### Survey<a id="paper1"></a>
- **WavChat: A Survey of Spoken Dialogue Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2411.13577)] [[github](https://github.com/jishengpeng/WavChat)]
- **Recent Advances in Speech Language Models: A Survey** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2410.03751)]
- **A Comprehensive Survey on Deep Multimodal Learning with Missing Modality** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.07825)]
- **A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.01319)]
- **MM-LLMs: Recent Advances in MultiModal Large Language Models** , 2024 , ACL [[paper](https://arxiv.org/pdf/2401.13601)] [[demo](https://mm-llms.github.io/)]
- **A Survey on Multimodal Large Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.13549)] [[github](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)]
- **Sparks of Large Audio Models: A Survey and Outlook** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.12792)]

### Speech/Audio-Extended Large Language Model<a id="paper2"></a>
#### Multi-modal
- **Baichuan-Omni Technical Report** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2410.08565)] [[github](https://github.com/westlake-baichuan-mllm/bc-omni)]
- **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context** , 2024 , arXiv [[paper](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)] 
- **The Llama 3 Herd of Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.21783)]
- **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2405.05945)] [[github](https://github.com/Alpha-VLLM/Lumina-T2X)]
- **VITA: Towards Open-Source Interactive Omni Multimodal LLM** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.05211)] [[demo](https://vita-home.github.io/)]
- **NExT-GPT: Any-to-Any Multimodal LLM** , 2024 , ICML [[paper](https://arxiv.org/pdf/2309.05519)] [[github](https://github.com/NExT-GPT/NExT-GPT)]
- **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** , 2024 , arXiv [[paper](https://arxiv.org/abs/2402.12226)] [[github](https://github.com/OpenMOSS/AnyGPT)]
- **Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.09093)] [[github](https://github.com/lyuchenyang/Macaw-LLM)]
#### Real-time interaction
- **Language Model Can Listen While Speaking** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.02622)] [[demo](http://ziyang.tech/LSLM/)]
- **SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2412.15649)] [[demo](https://slam-omni.github.io/)]
- **SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2411.18138)]
- **Advancing Speech Language Models by Scaling Supervised Fine-Tuning with Over 60,000 Hours of Synthetic Speech Dialogue Data** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2412.01078)]
- **Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2411.00774)] [[github](https://github.com/VITA-MLLM/Freeze-Omni)] [[demo](https://freeze-omni.github.io/)]
- **OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2410.17799)] [[demo](https://omniflatten.github.io/)]
- **Beyond the Turn-Based Game: Enabling Real-Time Conversations with Duplex Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.15718)] [[github](https://github.com/thunlp/duplex-model)]
- **Moshi: a speech-text foundation model for real-time dialogue** , 2024 [[paper](https://kyutai.org/Moshi.pdf)] [[github](https://github.com/kyutai-labs/moshi)]
- **LLaMA-Omni: Seamless Speech Interaction with Large Language Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.06666)] [[github](https://github.com/ictnlp/LLaMA-Omni)]
- **Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.16725)] [[github](https://github.com/gpt-omni/mini-omni)]
#### Speech/Audio understanding & generation
- **GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.11768)] [[github](https://github.com/Sreyan88/GAMA)] [[demo](https://sreyan88.github.io/gamaaudio/)]
- **WavLLM: Towards Robust and Adaptive Speech Large Language Model** , 2024 , EMNLP [[paper](https://arxiv.org/pdf/2404.00656)] [[github](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)]
- **EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.18042)] [[demo](https://emova-ollm.github.io/)]
- **MooER: LLM-based Speech Recognition and Translation Models from Moore Threads** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.05101)]  [[github](https://github.com/MooreThreads/MooER)]
- **Qwen2-Audio Technical Report** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.10759)] [[github](https://github.com/QwenLM/Qwen2-Audio)]
- **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2402.08846)] [[github](https://github.com/X-LANCE/SLAM-LLM)]
- **SALMONN: Towards Generic Hearing Abilities for Large Language Models** , 2024 , ICLR [[paper](https://openreview.net/pdf?id=14rn7HpKVk)] [[github](https://github.com/bytedance/SALMONN)]
- **LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2310.04673)] [[demo](https://lauragpt.github.io/)] 
- **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2311.07919)] [[github](https://github.com/QwenLM/Qwen-Audio)]
- **SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities** , 2023 , EMNLP [[paper](https://arxiv.org/pdf/2305.11000)] [[github](https://github.com/0nutation/SpeechGPT)]
- **Pengi: An Audio Language Model for Audio Tasks** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2305.11834)] [[github](https://github.com/microsoft/Pengi)]
- **AudioPaLM: A Large Language Model That Can Speak and Listen** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.12925)] [[demo](https://google-research.github.io/seanet/audiopalm/examples/)]

### Speech Recognition & Translation<a id="paper3"></a>
- **Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.04675)] [[demo](https://bytedancespeech.github.io/seedasr_tech_report/)]
- **OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2401.16658)] [[github](https://github.com/espnet/espnet/tree/master/egs2/owsm_v3.1/s2t1)]
- **Seamless: Multilingual Expressive and Streaming Speech Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2312.05187)] [[github](https://github.com/facebookresearch/seamless_communication)]
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.11596)] [[github](https://github.com/facebookresearch/seamless_communication)]
- Whisper: **Robust Speech Recognition via Large-Scale Weak Supervision** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.04356)] [[github](https://github.com/openai/whisper)]

### Speech Synthesis<a id="paper4"></a>
- **Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2411.01156)] [[github](https://github.com/fishaudio/fish-speech)]
- **F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2410.06885)] [[github](https://github.com/SWivid/F5-TTS)]
- **FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.03283)] [[github](https://github.com/FireRedTeam/FireRedTTS)] [[demo](https://fireredteam.github.io/demos/firered_tts/)]
- **MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.00750)] [[github](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)] [[demo](https://maskgct.github.io/)]
- **Natural language guidance of high-fidelity text-to-speech with synthetic annotations** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2402.01912)]  [[demo](https://www.text-description-to-speech.com/)]
- **CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.05407)] [[github](https://github.com/FunAudioLLM/CosyVoice)] [[demo](https://fun-audio-llm.github.io/)]
- **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.02430)] [[demo](https://bytedancespeech.github.io/seedtts_tech_report/)]
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2403.03100)] [[github](https://speechresearch.github.io/naturalspeech3/)]
- **Boosting Large Language Model for Speech Synthesis: An Empirical Study** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2401.00246)]
- **SoundStorm: Efficient Parallel Audio Generation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2305.09636)] [[demo](https://google-research.github.io/seanet/soundstorm/examples/)]
- VALL-E: **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.02111)] [[github](https://github.com/lifeiteng/vall-e)]
- **PromptTTS: Controllable Text-to-Speech with Text Descriptions** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2211.12171)] [[demo](https://speechresearch.github.io/prompttts/)]
- **FastSpeech 2: Fast and High-Quality End-to-End Text to Speech** , 2021 , ICLR [[paper](https://arxiv.org/pdf/2006.04558)] [[demo](https://speechresearch.github.io/fastspeech2/)]

### Self-supervised Pre-trained Speech Model<a id="paper5"></a>
- **BEATs: Audio Pre-Training with Acoustic Tokenizers** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.09058)] [[github](https://github.com/microsoft/unilm/tree/master/beats)]
- **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2202.03555)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
- **W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2108.06209)] 
- **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** , 2021 , arXiv [[paper](https://arxiv.org/abs/2110.13900)]
- **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2106.07447)]
- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** , 2020 , NeurIPS [[paper](https://arxiv.org/pdf/2006.11477)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]
- **wav2vec: Unsupervised Pre-Training for Speech Recognition** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1904.05862)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]

### Speech Tokenization/Discretization<a id="paper6"></a>
- **TS3-Codec: Transformer-Based Simple Streaming Single Codec** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2411.18803)]
- **Universal Speech Token Learning Via Low-Bitrate Neural Codec and Pretrained Representations** , 2024 , IEEE Journal of Selected Topics in Signal Processing [[paper](https://ieeexplore.ieee.org/abstract/document/10738376)]
- **Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis** , 2024 , ICLR [[paper](https://arxiv.org/abs/2306.00814)] [[github](https://github.com/tarepan/vocos-official)] [[demo](https://gemelo-ai.github.io/vocos/)]
- **WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.16532)] [[github](https://github.com/jishengpeng/WavTokenizer)] [[demo](https://wavtokenizer.github.io/)]
- **RepCodec: A Speech Representation Codec for Speech Tokenization** , 2024 , ACL [[paper](https://arxiv.org/pdf/2309.00169)] [[github](https://github.com/mct10/RepCodec)]
- **FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2309.07405)] [[github](https://github.com/modelscope/FunCodec)] [[demo](https://funcodec.github.io/)]
- **Single-Codec: Single-Codebook Speech Codec towards High-Performance Speech Generation** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2406.07422)] [[demo](https://kkksuper.github.io/Single-Codec/)]
- **ToneUnit: A Speech Discretization Approach for Tonal Language Speech Synthesis** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.08989)] [[demo](https://toneunit1225.github.io/)]
- **SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models** , 2024 , ICLR [[paper](https://arxiv.org/pdf/2308.16692)] [[github](https://github.com/ZhangXInFD/SpeechTokenizer)]
- **Finite Scalar Quantization: VQ-VAE Made Simple** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2309.15505)] [[github](https://github.com/google-research/google-research/tree/master/fsq)]
- **High-Fidelity Audio Compression with Improved RVQGAN** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2306.06546)] [[github](https://github.com/descriptinc/descript-audio-codec)]
- Encodec: **High Fidelity Neural Audio Compression** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2210.13438)] [[github](https://github.com/facebookresearch/encodec)]
- **SoundStream: An End-to-End Neural Audio Codec** , 2021 , TASLP [[paper](https://arxiv.org/pdf/2107.03312)] [[github](https://github.com/wesbz/SoundStream)] [[demo](https://google-research.github.io/seanet/soundstream/examples/)]
- **vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1910.05453)]
- VQ-VAE: **Neural Discrete Representation Learning** , 2017 , NeurIPS, [[paper](https://arxiv.org/pdf/1711.00937)] [[code](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)]

### Music Model<a id="paper7"></a>
- **Drop the beat! Freestyler for Accompaniment Conditioned Rapping Voice Generation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.15474)]
- **Seed-Music: A Unified Framework for High Quality and Controlled Music Generation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.09214)]
- **QA-MDT: Quality-aware Masked Diffusion Transformer for Enhanced Music Generation** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2405.15863)] [[github](https://github.com/ivcylc/qa-mdt)] [[github](https://qa-mdt.github.io/)]
- **M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2311.11255)] [[github](https://github.com/shansongliu/M2UGen)]
- **SingSong: Generating musical accompaniments from singing** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.12662)] [[example](https://storage.googleapis.com/sing-song/index.html)]
- **Simple and Controllable Music Generation** , 2023 , NeurIPS , [[paper](https://arxiv.org/pdf/2306.05284)] [[github](https://github.com/facebookresearch/audiocraft)]
- **MusicLM: Generating Music From Text** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.11325)] [[demo](https://google-research.github.io/seanet/musiclm/examples/)]
- **MuLan: A Joint Embedding of Music Audio and Natural Language** , 2022 , ISMIR [[paper](https://arxiv.org/pdf/2208.12415)] [[github](https://github.com/lucidrains/musiclm-pytorch)]

### Speech/Audio/Music Data<a id="paper8"></a>
- **Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation** , 2024 , SLT [[paper](https://arxiv.org/pdf/2407.05361)] [[huggingface](https://huggingface.co/datasets/amphion/Emilia-Dataset)]
- **WenetSpeech4TTS: A 12,800-hour Mandarin TTS Corpus for Large Speech Generation Model Benchmark** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2406.05763)]
- **MSR-86K: An Evolving, Multilingual Corpus with 86,300 Hours of Transcribed Audio for Speech Recognition Research** , 2024 , Interspeech [[paper](https://www.arxiv.org/pdf/2406.18301)] [[huggingface](https://huggingface.co/datasets/Alex-Song/MSR-86K)] 
- **YODAS: Youtube-Oriented Dataset for Audio and Speech** , 2023 , ASRU [[paper](https://arxiv.org/pdf/2406.00899)] [[demo](https://emilia-dataset.github.io/Emilia-Demo-Page/)]
- **GigaST: A 10,000-hour Pseudo Speech Translation Corpus** , 2023 , Interspeech [[paper](https://arxiv.org/pdf/2204.03939)] [[demo](https://st-benchmark.github.io/)]
- **AudioCaps: Generating Captions for Audios in The Wild** , 2019 , ACL [[paper](https://aclanthology.org/N19-1011.pdf)] [[demo](https://audiocaps.github.io/)]

### Benchmark<a id="paper9"></a>
- **MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2410.19168)] [[github](https://github.com/Sakshi113/MMAU)] [[demo](https://sakshi113.github.io/mmau_homepage/)]
- **AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension** , 2024 , ACL [[paper](https://aclanthology.org/2024.acl-long.109.pdf)] [[github](https://github.com/OFA-Sys/AIR-Bench)]
- **SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.13340)] [[github](https://github.com/amphionspace/SD-Eval)]

### Safety & Risk<a id="paper10"></a>
- **GPT-4o System Card** , 2024 [[paper](https://cdn.openai.com/gpt-4o-system-card.pdf)]
- **Safe Guard: an LLM-agent for Real-time Voice-based Hate Speech Detection in Social Virtual Reality** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2409.15623)]
- **SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models** , 2024 , ACL [[paper](https://aclanthology.org/2024.findings-acl.596.pdf)]
- **Warning: humans cannot reliably detect speech deepfakes** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2301.07829)]

### Others<a id="paper11"></a>
- **Low-latency Speech Enhancement via Speech Token Generation** , 2024 , ICASSP [[paper](https://arxiv.org/pdf/2310.08981)]
- **Genhancer: High-Fidelity Speech Enhancement via Generative Modeling on Discrete Codec Tokens** , 2024 , Interspeech [[paper](https://minjekim.com/wp-content/uploads/interspeech2024_hyang.pdf)]
- **Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2304.13731)] [[github](https://github.com/declare-lab/tango)] [[demo](https://tango-web.github.io/)]
- **AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models** , 2023 , NeurIPS [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/e1b619a9e241606a23eb21767f16cf81-Paper-Conference.pdf)] [[demo](https://audioldm.github.io/)]
- **AudioLM: a Language Modeling Approach to Audio Generation** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2209.03143)] [[demo](https://google-research.github.io/seanet/audiolm/examples/)]

## Open-source Large Speech/Audio Models<a id="model"></a>
- **Whisper**
  - [[github](https://github.com/openai/whisper)] ![](https://img.shields.io/github/stars/openai/whisper.svg)
  - Task: ASR
  - Institution: OpenAI 
  - Introduction: Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform **multilingual speech recognition**, **speech translation**, and **language identification**. It supports speech recognition for 100 languages. Five model sizes are trained, including tiny(39 M), base(74 M), small(244 M), medium(769 M), large(1550 M) and the newest large v3 model is trained on 5M hours of labeled data.
- **Seamless**
  - [[github](https://github.com/facebookresearch/seamless_communication)] ![](https://img.shields.io/github/stars/facebookresearch/seamless_communication.svg)
  - Task: ASR/S2TT/S2ST/T2TT/T2ST
  - Institution: Meta 
  - Introduction: Seamless is a family of AI models that enable more natural and authentic communication across languages. It contains three main models, SeamlessM4T, SeamlessExpressive and SeamlessStreaming. **SeamlessM4T** is a massive multilingual multimodal machine translation model supporting around 100 languages. SeamlessM4T serves as foundation for **SeamlessExpressive**, a model that preserves elements of prosody and voice style across languages and **SeamlessStreaming**, a model supporting simultaneous translation and streaming ASR for around 100 languages. SeamlessExpressive and SeamlessStreaming are combined into Seamless, a unified model featuring multilinguality, real-time and expressive translations.
- **SenseVoice**
  - [[github](https://github.com/FunAudioLLM/SenseVoice)] ![](https://img.shields.io/github/stars/FunAudioLLM/SenseVoice.svg)
  - Task: ASR 
  - Institution: Alibaba 
  - Introduction: SenseVoice is a speech foundation model with multiple speech understanding capabilities, including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED).
- **CosyVoice**
  - [[github](https://github.com/FunAudioLLM/CosyVoice)] ![](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice.svg)
  - Task: TTS
  - Institution: Alibaba 
- **Qwen2-Audio**
  - [[github](https://github.com/QwenLM/Qwen2-Audio)] ![](https://img.shields.io/github/stars/QwenLM/Qwen2-Audio.svg)
  - Task: S2T(Voice Chat & Audio Analysis)
  - Institution: Alibaba 
  - Introduction: The latest progress of Qwen-Audio, a large-scale audio-language model called Qwen2-Audio, which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. Two distinct audio interaction modes are introductioned: (1) voice chat: users can freely engage in voice interactions with Qwen2-Audio without text input; (2) audio analysis: users could provide audio and text instructions for analysis during the interaction; Two models are released of the Qwen2-Audio series: Qwen2-Audio-7B and Qwen2-Audio-7B-Instruct.
- **SALMONN**
  - [[github](https://github.com/bytedance/SALMONN)] ![](https://img.shields.io/github/stars/bytedance/SALMONN.svg)
  - Task: S2T(General Audio Proceessing)
  - Institution: Tsinghua University and ByteDance 
  - Introduction: SALMONN is a large language model (LLM) enabling speech, audio events, and music inputs. Instead of speech-only input or audio-event-only input, SALMONN can perceive and understand all kinds of audio inputs and therefore obtain emerging capabilities such as multilingual speech recognition and translation and audio-speech co-reasoning.
- **Reverb**
  - [[github](https://github.com/revdotcom/reverb)] ![](https://img.shields.io/github/stars/revdotcom/reverb.svg)
  - Task: ASR & Speech Diarization
  - Institution: Rev 
  - Introduction: Open source inference and evaluation code for Rev's state-of-the-art speech recognition and diarization models. The speech recognition (ASR) code uses the WeNet framework and the speech diarization code uses the Pyannote framework. Reverb ASR was trained on **200,000 hours** of English speech, all expertly **transcribed by humans**. The speech recognition models released outperform all existing open source speech recognition models across a variety of long-form speech recognition domains.
- **NVIDIA NeMo ASR**
  - [[github](https://github.com/NVIDIA/NeMo)] ![](https://img.shields.io/github/stars/NVIDIA/NeMo.svg)
  - Task: ASR
  - Institution: NVIDIA 
  - Introduction: NVIDIA NeMo team released a number of inference optimizations for CTC, RNN-T, and TDT models that resulted in up to 10x inference speed-up. These models now exceed an inverse real-time factor (RTFx) of 2,000, with some reaching RTFx of even 6,000.
- **Westlake-Omni**
  - [[github](https://github.com/xinchen-ai/Westlake-Omni)] ![](https://img.shields.io/github/stars/xinchen-ai/Westlake-Omni.svg)
  - Task: S2S(Speech Interaction)
  - Institution: xinchen-ai 
  - Introduction: Westlake-Omni is an open-source Chinese emotional speech interaction large language model that utilizes discrete representations to achieve unified processing of speech and text modalities. The model supports low-latency generation and high-quality Chinese emotional speech interaction.
- **GLM-4-Voice**
  - [[github](https://github.com/THUDM/GLM-4-Voice)] ![](https://img.shields.io/github/stars/THUDM/GLM-4-Voice.svg)
  - Task: S2S(Speech Interaction)
  - Institution: Zhipu AI 
  - Introduction: GLM-4-Voice is an end-to-end voice model launched by Zhipu AI. GLM-4-Voice can directly understand and generate Chinese and English speech, engage in real-time voice conversations, and change attributes such as emotion, intonation, speech rate, and dialect based on user instructions.
- **MiniCPM-o**
  - [[github](https://github.com/OpenBMB/MiniCPM-o)] ![](https://img.shields.io/github/stars/OpenBMB/MiniCPM-o.svg)
  - Task: S2S(Multi-modal LLM)
  - Institution: THUNLP & Modelbest
  - Introduction: MiniCPM-o is the latest series of end-side multimodal LLMs (MLLMs) ungraded from MiniCPM-V. The models can now take image, video, text, and audio as inputs and provide high-quality text and speech outputs in an end-to-end fashion.

## Datasets<a id="datasets"></a>
Only common benchmarks and large-scale training datasets are listed here.
- Speech Recognition & Translation
- Speech Synthesis
- Audio Caption
- Audio Question Answering

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
| fleurs [[download](https://huggingface.co/datasets/google/fleurs)] | Multilingual | ASR/ST/MT | 10 |
| AudioCaps [[download](https://huggingface.co/datasets/d0rj/audiocaps)] | - | AC | - |
| VoiceAssistant-400K [[download](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K)] | English | AQA | - |

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
- **fish-speech**: Input a 10 to 30-second vocal sample to generate high-quality TTS output. [[github](https://github.com/fishaudio/fish-speech)]

## Applications<a id="application"></a>
- **GPT-4o** [[link](https://openai.com/index/hello-gpt-4o/)]
- **Moshi** [[link](https://moshi.chat/)]
- **heygen** [[link](https://www.heygen.com/)]
- **Hume AI** [[link](https://www.hume.ai/)]
- **Suno** [[link](https://suno.com/)]
- **Google Illuminate** [[link](https://illuminate.google.com/home?pli=1)]
- **ElevenLabs** [[link](https://elevenlabs.io/)]
