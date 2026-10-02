# Automatic mastering and restoration of stereo mixes: literature, runnable baselines, datasets, evaluation practice

Compiled 2026-10-02. Scope: prior work and locally runnable baselines for a system that restores and masters finished stereo mixes.

## How to read this file

Verification tags after each citation say where the bibliographic record was checked:

- `[arXiv]` arXiv API record (title, authors, date, comment field).
- `[Crossref]` Crossref API record (DOI, venue, pages).
- `[OpenAlex]` OpenAlex record (used for AES convention and workshop papers that have no DOI).
- `[repo]` GitHub or Hugging Face API, README, LICENSE file.
- `[full text]` numbers were read from the paper PDF text.
- `[abstract]` only the abstract was read; details beyond it are not asserted.
- `[snippet]` a search-engine summary of the publisher page. Treat as weaker evidence.
- `[UNVERIFIED]` could not be confirmed from a primary source.

Known access limits on the day of compilation: the AES E-Library (aes.org) sits behind a bot wall and returned 403 to every fetch, so AES convention papers are verified through OpenAlex, institutional repositories, or search snippets, and their paper numbers carry a tag when only a snippet confirmed them. The University of Salford repository and Fraunhofer Publica also blocked fetches. OpenAlex returned wrong abstracts for the two Wilson and Fazenda JAES papers, so those summaries rely on snippets of the Salford repository pages.

---

## A. Academic literature (2010 to 2026)

### A1. Automatic and intelligent mastering

**Mimilakis, Drossos, Virtanen, Schuller (2016). "Deep Neural Networks for Dynamic Range Compression in Mastering Applications." AES 140th Convention, Paris, Paper 9539.** AES E-Library 18237. No DOI. `[OpenAlex]` `[snippet]` for paper number and abstract.
- Method: a DNN predicts frequency-dependent gain coefficients that implement dynamic range compression in the time-frequency domain, using magnitude spectra from a critical-band filter bank.
- Data: not confirmed (full text not accessed) `[UNVERIFIED]`.
- Evaluation: listening tests with professional music producers and mastering engineers.
- Headline: the abstract reports performance equivalent on average to professionally mastered audio, and improvements over "relevant and commercial software". No numbers confirmed.
- Code: none found.

**Mimilakis, Drossos, Floros, Katerelos (2013). "Automated Tonal Balance Enhancement for Audio Mastering Applications." AES 134th Convention, Rome.** AES E-Library 16737. `[OpenAlex]`; paper number `[UNVERIFIED]`.
- Method: adaptive, automated equalization for mastering driven by fundamental frequency tracking `[snippet]`.
- Evaluation: PEAQ plus subjective listening tests `[snippet]`. Numbers not confirmed.

**Ma, Reiss, Black (2013). "Implementation of an Intelligent Equalization Tool Using Yule-Walker for Music Mixing and Mastering." AES 134th Convention, Rome.** AES E-Library 16792. `[OpenAlex]`; paper number `[UNVERIFIED]`.
- Method per title: target-curve equalization with filters designed by the Yule-Walker method. Abstract not accessed, so data and evaluation are `[UNVERIFIED]`.

**Martinez Ramirez, Wang, Smaragdis, Bryan (2021). "Differentiable Signal Processing With Black-Box Audio Effects" (DeepAFx). ICASSP 2021, pp. 66-70.** DOI 10.1109/ICASSP39728.2021.9415103, arXiv:2105.04752. `[Crossref]` `[arXiv]` `[full text]`.
- Method: a deep encoder predicts parameters of non-differentiable third-party plugins (LV2); gradients are approximated with a stochastic scheme so the plugin sits inside the training graph. One of three tasks is automatic music mastering with a multiband compressor, a 32-band graphic EQ, and a limiter in series (50 parameters).
- Data (mastering task): 138 unmastered and mastered track pairs collected from the Mixing Secrets library, time-aligned by cross-correlation, unmastered inputs normalized to -25 dBFS; 429.3 / 51.1 / 50.3 minutes train / val / test; 22,050 Hz, mono.
- Evaluation: MFCC cosine distance as objective proxy. MUSHRA via the Web Audio Evaluation Tool, 17 participants (musicians, engineers, critical listeners), five 4-second samples per task, post-hoc paired t-tests with Bonferroni correction. Baseline for mastering: an online mastering service cited as LANDR.
- Headline: mastering medians were low anchor 0.15, mid anchor 0.24, hidden reference 0.95, proposed 0.88, online service 0.79. The proposed-vs-service difference had p about 0.0378, which misses the paper's p < 0.01 threshold.
- Code: https://github.com/adobe-research/DeepAFx (TensorFlow 2.2, Docker, LV2). The README advertises no pretrained mastering checkpoint. The 138 pairs were not released.

**Koo, Paik, Lee (2022). "End-to-end Music Remastering System Using Self-supervised and Adversarial Training." ICASSP 2022, pp. 4608-4612.** DOI 10.1109/ICASSP43922.2022.9746389, arXiv:2202.08520. `[Crossref]` `[arXiv]` `[full text]`.
- Method: a contrastively trained Music Effects Encoder embeds the mastering style of a reference; a Mastering Cloner transforms the input waveform toward that style; a projection discriminator adds an adversarial term. Training is self-supervised: two segments of one song get the same random mastering manipulation.
- Data: encoder on MTG-Jamendo (55k+ stereo tracks, 44.1 kHz). Cloner on a private collection of already-mastered pop, rock and hip-hop songs released after 2000 (760 train, 28 validation).
- Evaluation: RMS difference, side-channel RMS difference, frequency-weighted segmental SNR, STOI; qualitative inspection with Logic Pro Multimeter and Ozone 9 Imager. MUSHRA via Web Audio Evaluation Tool with 17 musicians and sound engineers, 15 questions, paired t-tests with Bonferroni correction.
- Headline: best model reduced RMS difference from 0.0319 (input) to 0.0212 and side RMS difference from 0.0531 to 0.0396. Listening-test medians were 0.32, 0.67, 0.71 and 1.00 (order inferred from the text: low anchor, unprocessed input, proposed, hidden reference); proposed vs input differed with p < 0.001. The authors note tone and timbre matched less well than loudness and width.
- Code: https://github.com/jhtonyKoo/e2e_music_remastering_system (MIT). Checkpoints on Google Drive, publicly downloadable without login (checked 2026-10-02).

**Koo, Martinez-Ramirez, Liao, Fabbro, Mancusi, Mitsufuji (2025). "ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors." ISMIR 2025.** arXiv:2506.16889. `[arXiv]` `[full text]`.
- Method: reference-based mastering style transfer. A TCN "Mastering Style Converter" (10.5 M parameters) is conditioned on an FXencoder embedding of the reference. Two variants: black-box (direct waveform) and white-box (predicts parameters of a differentiable mastering chain built with dasp-pytorch and torchcomp, with six modules: 6-band parametric EQ, distortion, 3-band compressor, makeup gain, stereo imager, limiter). Inference-time optimization (ITO) refines the reference embedding against an audio-feature loss or a CLAP text/audio loss for up to 100 steps.
- Data: trained on MoisesDB mixtures and validated on MUSDB18 mixtures because they are "not fully mastered"; random effect manipulation creates synthetic masters, with Fx-Normalization of EQ, stereo image and loudness. Evaluation on 200 MTG-Jamendo songs (100 inputs, 100 references), 30-second excerpts. Training segments are 11.8 s.
- Evaluation: audio feature (AF) loss, a new Dynamic Range Variability (DRV) measure, FXencoder cosine similarity, and FAD with CLAP, DAC and EnCodec embeddings against a Jamendo subset. Baselines: Fx-Normalization feature matching, Matchering, and the 2022 E2E Remastering model. MUSHRA-type test with 10 participants (2 to 5 years of recording, mixing or mastering experience), 8 questions, 30-second stimuli, input as low anchor, no high anchor; pairwise t-tests.
- Headline (Table 1): Matchering AF 0.160 / DRV 0.823 / FAD-CLAP 110.8 / FAD-DAC 126.1 / FAD-EnCodec 59.34. E2E Remastering 0.288 / 0.858 / 104.3 / 176.7 / 37.19. White-box with trained encoder 0.186 / 0.521 / 93.2 / 101.4 / 38.90; adding ITO gives 0.139 / 0.474 / 105.2 / 109.1 / 42.99. ITO improves feature matching and trades away some FAD. Proposed systems beat baselines in the listening test (p < 0.05).
- Code: https://github.com/SonyResearch/ITO-Master, license CC BY-NC 4.0, `pip install ito_master`. Weights download automatically from the public Hugging Face Space `jhtonyKoo/ITO-Master`.

**Melechovsky, Mehrish, Roy, Herremans (2026). "SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering." ICML 2026 (PMLR 306), Seoul.** arXiv:2508.03448 (v5). `[arXiv]` `[full text]`.
- Method: a rectified-flow model (MM-DiT plus DiT blocks) operating in the latent space of the Stable Audio Open VAE at 44.1 kHz stereo, conditioned on a text instruction, with an automatic mode when no prompt is given. One model covers 19 degradations in five groups: EQ, dynamics, reverb, amplitude (including clipping), stereo. The authors define "mastering" as targeted perceptual enhancement of degraded music and limit the scope to single songs.
- Data: SonicMaster dataset. About 580k Jamendo recordings filtered with Audiobox Aesthetics to 2,500 songs per genre group across 10 groups (about 25k 30-second clips), each corrupted with one to three effects and paired with a text prompt. Test set: 7k degraded clips against 1k clean references.
- Evaluation: FAD on CLAP embeddings, KL divergence, SSIM on 128-bin mel spectrograms, Audiobox Aesthetics Production Quality (PQ), plus degradation-specific feature errors (band energy ratios, RMS-frame standard deviation, onset strength, modulation-spectrum distance, spectral flatness, mid/side RMS ratio). Baselines: Text2FX (two variants), WPE and HPSS dereverberation, Mel2Mel plus DiffWave, and VAE reconstruction of the input. Listening study 1: 12 listeners (7 music experts, 5 MIR researchers), 43 input/output pairs, 7-point Likert scales, paired t-tests. Listening study 2: 20 participants, forced preference on 10 EQ and 10 reverb samples.
- Headline: output quality rated above input in every category, significant everywhere except EQ. In study 2 SonicMaster was chosen in 191 of 200 reverb comparisons and 180 of 200 EQ comparisons. The paper states that FAD after processing is marginally higher than for the degraded input and that SSIM drops below the degraded input, so the gains show in PQ, KL and the degradation-specific errors. On the RemFX test split it reports SI-SDR of 47.11 dB (dynamics) and 45.76 dB (reverb) against best baselines of 20.08 and 13.59 dB.
- Independent check: Karystinaios et al. (arXiv:2607.12872) benchmark SonicMaster as an offline reference and find it strongest on SonicMaster-set FAD and KL but weaker on signal-preservation metrics (SI-SNR family).
- Code: https://github.com/AMAAI-Lab/SonicMaster (Apache-2.0). Weights at `amaai-lab/SonicMaster` on Hugging Face, ungated (3.45 GB). Inference also loads the VAE from `stabilityai/stable-audio-open-1.0`, which is a gated repository (see section B).

**Mourgela, Quinton, Bissas, Reiss, Ronan (2024). "Exploring trends in audio mixes and masters: Insights from a dataset analysis." AES 157th Convention, New York.** arXiv:2412.03373. `[arXiv]` `[full text]`.
- Method: descriptive analysis of metrics logged by the MixCheck Studio web platform (RoEx): integrated loudness, mono compatibility, clipping, phase, compression, tonal profile across 30 user-specified genres.
- Data: 67,838 tracks labelled by users as mixes and 150,217 as masters.
- Headline: about 79% of masters exceed -14 LUFS; only 42.53% of masters show no clipping; mixes peak around -23 LUFS.
- Code/data: audio not released as far as verified.

**Elliott, Chon (2022). "A Comparative Study of Music Mastered by Human Engineers and Automated Services." JAES 70(9), 764-776.** DOI 10.17743/jaes.2022.0050. `[Crossref]`; method and results `[snippet]`.
- Method: tracks mastered by two human engineers and two automated services; listeners tried to identify the human master and ranked preference; timbral and spectral features compared.
- Headline: listeners could not consistently identify the human masters; a preference for human masters appeared for jazz excerpts and was absent for rock. Listener count and service names `[UNVERIFIED]`.

**Piotrowska, Piotrowski, Kostek (2017). "A Study on Audio Signal Processed by 'Instant Mastering' Services." AES 142nd Convention, Berlin, Paper 9719.** AES E-Library 18597. No DOI. `[snippet]`.
- Method: 10 songs and laboratory signals sent through eight online mastering services, several times each, with timbre, dynamics and loudness descriptors computed before and after. Objective only.

Critical and qualitative studies of automated mastering (no listening tests): Sterne and Razlogova 2019 (DOI 10.1177/2056305119847525) and 2021 (DOI 10.1080/09502386.2021.1895247); Birtchnell and Elliott 2018 (DOI 10.1016/j.geoforum.2018.08.005); Birtchnell 2018 (DOI 10.1177/2053951718808553); Collins, Renzo, Keith, Mesker 2021 (DOI 10.1080/03007766.2019.1699339). All `[Crossref]`.

### A2. Audio effect style transfer and differentiable effects

**Steinmetz, Bryan, Reiss (2022). "Style Transfer of Audio Effects with Differentiable Signal Processing" (DeepAFx-ST). JAES 70(9), 708-721.** DOI 10.17743/jaes.2022.0025, arXiv:2207.08759. `[Crossref]` `[arXiv]` `[full text]`.
- Method: an encoder analyses input and reference and predicts parameters of a parametric EQ followed by a compressor. Training is self-supervised: two differently processed versions of one recording act as input and target. The paper compares neural proxies, SPSA gradient approximation, and automatic differentiation of DSP implementations. The text names music mastering as an application of this EQ plus dynamics chain.
- Data: LibriTTS (360 h, 24 kHz) and MTG-Jamendo (55k+ songs, downmixed to mono, 24 kHz), 90/5/5 splits. Generalization tested on DAPS, VCTK, MUSDB18 and at 44.1 kHz.
- Evaluation: PESQ, multi-resolution STFT error, mel-spectral distance (MSD), spectral centroid error (SCE), RMS error, LUFS error; a style-classification probe on the learned embeddings. No formal listening test.
- Headline: autodiff DSP performs best overall; neural proxies degrade when moved from 24 kHz to 44.1 kHz.
- Code: https://github.com/adobe-research/DeepAFx-ST, Adobe Research License (noncommercial research only). Checkpoints in a GitHub release (1.09 GB, no login).

**Steinmetz, Singh, Comunita, Ibnyahya, Yuan, Benetos, Reiss (2024). "ST-ITO: Controlling Audio Effects for Style Transfer with Inference-Time Optimization." ISMIR 2024.** arXiv:2410.21233. `[arXiv]` `[full text]`.
- Method: a self-supervised effects representation (AFx-Rep) trained on a pretext task that predicts which effect and which preset were applied; at inference a gradient-free search (evolution strategy) optimises the parameters of arbitrary, non-differentiable effects (VST plugins, pedalboard) to match the reference in that embedding space.
- Data: about 60 hours of audio drawn from several sets including MTG-Jamendo and ENST-Drums, 48 kHz; real-world test cases with vocals, music and speech (DAPS, MUSDB18).
- Evaluation: zero-shot style classification accuracy; multi-stimulus listening study with 23 participants experienced in audio engineering, 0 to 100 similarity, ten test cases.
- Headline: at least on par with an extended DeepAFx-ST, better for spatial and delay styles; a rule-based baseline wins for simple one-filter styles.
- Code: https://github.com/csteinmetz1/st-ito (Apache-2.0).

**Mimilakis, Bryan, Smaragdis (2020). "One-Shot Parametric Audio Production Style Transfer with Application to Frequency Equalization." ICASSP 2020, pp. 256-260.** DOI 10.1109/ICASSP40776.2020.9054108. `[Crossref]`. Predicts parametric EQ settings from a single reference example.

**Colonel, Reiss (2021). "Reverse engineering of a recording mix with differentiable digital signal processing." JASA 150(1), 608-619.** DOI 10.1121/10.0005622. `[Crossref]`. Recovers gain, pan, EQ, compression and reverb settings from stems plus target mix by gradient descent.

Related tooling: dasp-pytorch (https://github.com/csteinmetz1/dasp-pytorch, Apache-2.0) supplies the differentiable EQ, compressor and reverb used by ITO-Master, Diff-MST, Text2FX and LLM2Fx. NablAFx (Comunita, Steinmetz, Reiss, arXiv:2502.11668) covers differentiable black-box and gray-box effect modelling. `[arXiv]` `[repo]`.

### A3. Automatic mixing

**Steinmetz, Pons, Pascual, Serra (2021). "Automatic multitrack mixing with a differentiable mixing console of neural audio effects." ICASSP 2021, pp. 71-75.** DOI 10.1109/ICASSP39728.2021.9414364, arXiv:2010.10291. `[Crossref]` `[arXiv]` `[full text]`.
- Method: per-channel neural proxies of a channel strip (EQ, compressor, reverb) with a permutation-invariant controller; trained on waveforms with a stereo-invariant multi-resolution STFT loss.
- Data: ENST-Drums (about 3 h) and MedleyDB (196 songs, about 7 h; 120 songs with at most 16 inputs, 65 with at most 6).
- Evaluation: APE-style test in the Web Audio Evaluation Tool with 16 audio engineers; Kruskal-Wallis H-test.
- Headline: near the human target on some passages and clearly worse on others.
- Code: https://github.com/csteinmetz1/automix-toolkit (Apache-2.0), checkpoints on Hugging Face, ungated.

**Martinez-Ramirez, Liao, Fabbro, Uhlich, Nagashima, Mitsufuji (2022). "Automatic music mixing with deep learning and out-of-domain data" (FxNorm-Automix). ISMIR 2022.** arXiv:2208.11428. `[arXiv]` `[full text]`.
- Method: trains on already-processed (wet) stems by first normalizing their loudness, EQ, panning, compression and reverb to dataset averages ("effect normalization"), then learning to map normalized stems to the mix.
- Data: MUSDB18 (86 / 14 / 50) and a private set of 1,505 additional multitracks; private test set of 18 dry multitrack songs.
- Evaluation: audio-feature distances; APE-style listening test restricted to professionals: 14 engineers with a mean of 11.6 years of mixing experience, six songs, six mixes each, 25-second excerpts, rated for Production Value, Clarity and Excitement, loudness matched to -23 dBFS, dry stems given as reference, no low anchor.
- Headline: for Production Value, some proposed models showed no statistically significant difference from human mixes. The authors observe that experienced engineers rate no mix as very good.
- Code: https://github.com/sony/FxNorm-automix (MIT), trained models stored in the repository; impulse responses used in the paper are not distributed.

**Koo, Martinez-Ramirez, Liao, Uhlich, Lee, Mitsufuji (2023). "Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects." ICASSP 2023.** DOI 10.1109/ICASSP49357.2023.10096458, arXiv:2211.02247. `[Crossref]` `[arXiv]` `[full text]`.
- Method: FXencoder, a contrastive encoder that separates effect style from content, and MixFXcloner, which converts stems to a reference mixing style.
- Data: MUSDB18 (86 / 14 / 50).
- Evaluation: listening test in the Web Audio Evaluation Tool with 11 audio engineers (mean 7.3 years), 12 questions; paired t-tests with Bonferroni correction.
- Mixture-level pipeline: the paper also evaluates inference on finished mixes. Input and reference mixtures are separated with Hybrid Demucs, each stem is converted, and the stems are summed. Mean SDR against the clean-stem result was 8.004 dB at mixture level and 4.554 dB stem-wise, and the style metric was almost unchanged. This is the earliest published separation-assisted per-stem effect transfer on stereo mixes that I found.
- Code: https://github.com/jhtonyKoo/music_mixing_style_transfer (MIT), checkpoints on Google Drive (public). The released inference script runs Demucs (`mdx_extra`) on the input and reference by default. FXencoder is the embedding used later by ITO-Master.

**Vanka, Steinmetz, Rolland, Reiss, Fazekas (2024). "Diff-MST: Differentiable Mixing Style Transfer." ISMIR 2024.** arXiv:2407.08889. `[arXiv]` `[full text]`.
- Method: a transformer controller predicts parameters of a differentiable mixing console (gain, pan, EQ, compressor) from raw tracks and a reference mix.
- Data: MedleyDB (196 songs) and Cambridge-MT (535 songs) multitracks, MTG-Jamendo songs as references.
- Evaluation: audio-feature loss terms (RMS, crest factor, stereo width, stereo imbalance, bark spectrum) and FAD (VGGish). The paper states that no listening test was run.
- Code: https://github.com/sai-soum/Diff-MST (CC BY-NC-SA 4.0). No checkpoint is published; issue #27 asking for one is open. Diff-MSTC (arXiv:2411.06576) is a Cubase prototype.

**Moliner, Martinez-Ramirez, Koo, Liao, Cheuk, Serra, Valimaki, Mitsufuji (2026). "Automatic Music Mixing Using a Generative Model of Effect Embeddings" (MEGAMI). ICASSP 2026, pp. 14582-14586.** DOI 10.1109/ICASSP55912.2026.11462677, arXiv:2511.08040. `[Crossref]` `[arXiv]` `[full text]`.
- Method: a diffusion model generates per-track effect embeddings conditioned on the dry tracks; a track-agnostic processor applies them. Models mixing as a one-to-many problem.
- Data: internal sets (about 20k professionally mixed songs with wet stems), a public set from MoisesDB and MedleyDB (248 multitracks), public dry sources; internal benchmark of 59 multitracks, 590 segments of 11.9 s; 44.1 kHz stereo.
- Evaluation: distributional metrics (KAD); multi-stimulus listening test in isolated booths, 12 volunteers (6 with production experience), seven songs presented twice, five stimuli per page.
- Code: https://github.com/SonyResearch/MEGAMI (CC BY-NC-SA 4.0), public checkpoints in a GitHub release.

2026 follow-ups, `[arXiv]` and partial `[full text]`:
- Zong, Shi, Reiss. "Diff2Mix: Controllable Music Mixing via Diffusion Models and Differentiable Audio Effects." ISMIR 2026. arXiv:2608.05442. KAD; webMUSHRA with 20 participants with professional mixing experience; pairwise Wilcoxon tests with Holm adjustment.
- Yeh, Chan, Hung, Yang. "Rethinking Automatic Music Mixing as Sequential Stem Blending." arXiv:2608.05506. Latent flow matching; KAD with CLAP and Fx-Encoder++ embeddings, tonal balance, RMS, crest factor, Audiobox PQ; MUSHRA-style test with 18 participants.
- Liu, Lin, Wei, Yan. "Beyond Dry References: Learning Relative Audio Effects Representations via Contrastive Distance Learning" (RelFx). ISMIR 2026. arXiv:2608.10573. `[abstract]`.
- Yu et al. "DiffVox." DAFx 2025. arXiv:2504.14735. Differentiable vocal effects chain fitted to professional mixes.

Classic knowledge-engineered and early machine-learning mixing work: Ma, De Man, Pestana, Black, Reiss, "Intelligent Multitrack Dynamic Range Compression," JAES 2015, pp. 412-426, DOI 10.17743/jaes.2015.0053 (Crossref returns no volume; multiple-stimulus test against a no-compression mix, two human mixes and an alternative approach) `[Crossref]` `[abstract]`; Hafezi and Reiss, "Autonomous Multitrack Equalization Based on Masking Reduction," JAES 63(5), 312-323, DOI 10.17743/jaes.2015.0021 `[Crossref]`; Martinez Ramirez, Stoller, Moffat, "A Deep Learning Approach to Intelligent Drum Mixing With the Wave-U-Net," JAES 69(3), 142-151, DOI 10.17743/jaes.2020.0031 `[Crossref]`; Mockenhaupt, Rieber, Nercessian, "Automatic Equalization for Individual Instrument Tracks Using Convolutional Neural Networks," DAFx 2024, arXiv:2407.16691 `[arXiv]`.

Surveys and books: De Man, Stables, Reiss, *Intelligent Music Production*, Routledge 2019, DOI 10.4324/9781315166100 `[Crossref]`; De Man, Reiss, Stables, "Ten Years of Automatic Mixing," 3rd Workshop on Intelligent Music Production, 2017 `[OpenAlex]`; Moffat and Sandler, "Approaches in Intelligent Music Production," Arts 8(4), 125, DOI 10.3390/arts8040125 `[Crossref]`; Pestana, PhD thesis "Automatic Mixing Systems Using Adaptive Digital Audio Effects," Universidade Catolica Portuguesa, 2013, http://hdl.handle.net/10400.14/10887 `[snippet]`; Pestana and Reiss, "Intelligent Audio Production Strategies Informed by Best Practices," 2014 (AES 53rd Conference venue `[UNVERIFIED]`, record at QMRO handle 123456789/11599) `[OpenAlex]`.

Practitioner studies: Vanka, Safi, Rolland, Fazekas, "Adoption of AI Technology in the Music Mixing Workflow: An Investigation," AES 154th Convention, Paper 10653, arXiv:2304.03407; Vanka et al., "The Role of Communication and Reference Songs in the Mixing Process," arXiv:2309.03404. `[arXiv]`.

### A4. Blind estimation and removal of audio effects

**Jeon, Lee (2023). "Music De-limiter Networks via Sample-wise Gain Inversion." WASPAA 2023.** DOI 10.1109/WASPAA58266.2023.10248055, arXiv:2308.01187. `[Crossref]` `[arXiv]` `[full text]`.
- Method: the network estimates the sample-wise gain a limiter applied and inverts it (SGI), instead of synthesising or masking the waveform.
- Data: musdb-XL-train, 300,000 four-second segments of randomly mixed MUSDB18-HQ stems processed with the iZotope Ozone 9 Maximizer, plus limited versions of the 100 training songs. Test: musdb-XL against MUSDB18-HQ.
- Evaluation: SI-SDR, multi-resolution spectrogram MSE, PEAQ, FAD, parameter count, MACs; dynamics descriptors (RMS, crest factor, dynamic complexity, LRA, spectral centroid); everything measured after loudness normalization to -14 LUFS with pyloudnorm.
- Headline: 24.0 dB SI-SDR reconstructing MUSDB18-HQ from musdb-XL. The paper documents an FAD failure case: a model with channel-polarity errors scored FAD 2.670 while SI-SDR looked fine, because the FAD pipeline downmixes to 16 kHz mono.
- Code: https://github.com/jeonchangbin49/De-limiter (MIT), 9 MB weights inside the repository.

**Jeon, Lee (2022). "Towards robust music source separation on loud commercial music." ISMIR 2022.** arXiv:2208.14355. `[arXiv]` `[full text]`.
- Contribution relevant here: loudness statistics showing the gap between research data and releases. Mean integrated loudness: MUSDB18-HQ test -15.92 LUFS (std 1.27), musdb-L -10.89 (1.19), musdb-XL -8.61 (1.17), 50 commercial songs -8.05 (1.06).
- Data release: musdb-L and musdb-XL as limiter gain ratios (Zenodo 7041331, CC BY 4.0).

**Rice, Steinmetz, Fazekas, Reiss (2023). "General Purpose Audio Effect Removal" (RemFX). WASPAA 2023.** DOI 10.1109/WASPAA58266.2023.10248157, arXiv:2308.16177. `[Crossref]` `[arXiv]` `[full text]`.
- Method: an effect classifier selects and chains effect-specific removal networks (distortion, compression, reverb, chorus, delay).
- Data: VocalSet, GuitarSet, DSD100 and IDMT-SMT-Drums sources, 48 kHz, about 5.5-second chunks at -20 LUFS, up to five simultaneous effects.
- Evaluation: SI-SDR and multi-resolution STFT error. No listening test.
- Headline: compositional removal beats monolithic models for few effects; all methods degrade with four or five effects.
- Code: https://github.com/mhrice/RemFx (Apache-2.0); checkpoints on Zenodo record 8218621 (open access, noncommercial license id).

**Imort, Fabbro, Martinez Ramirez, Uhlich, Koyama, Mitsufuji (2022). "Distortion Audio Effects: Learning How to Recover the Clean Signal." ISMIR 2022.** arXiv:2202.01664. `[arXiv]`. Guitar distortion and clipping removal with separation-style networks.

**Peladeau, Peeters (2024). "Blind Estimation of Audio Effects Using an Auto-Encoder Approach and Differentiable Digital Signal Processing." ICASSP 2024, pp. 856-860.** DOI 10.1109/ICASSP48485.2024.10448301, arXiv:2310.11781. `[Crossref]` `[abstract]`. Estimates parameters of "commonly used mastering AFXs" from the processed signal alone by optimising an audio-domain loss through differentiable or proxy effects; reports better audio match than parameter-loss training even when parameter error is higher.

**Lee, Park, Paik, Lee (2023). "Blind Estimation of Audio Processing Graph." ICASSP 2023.** DOI 10.1109/ICASSP49357.2023.10096581, arXiv:2303.08610. **Lee et al. (2024). "Searching For Music Mixing Graphs: A Pruning Approach." DAFx 2024.** arXiv:2406.01049. **Lee et al. (2025). "Reverse Engineering of Music Mixing Graphs With Differentiable Processors and Iterative Pruning." JAES 73(6), 344-365.** DOI 10.17743/jaes.2022.0212. `[Crossref]` `[arXiv]` `[abstract]`. Recover processing graphs from dry/mix pairs.

**Sun, Fourer, Maaref.** "Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics," arXiv:2411.04337; "Black-Box Optimization for Identifying and Inverting Audio Dynamic Range Control Effects," arXiv:2607.19645. `[arXiv]` `[abstract]`. Blind compressor parameter estimation and inversion.

**Hinrichs, Gerkens, Lange, Ostermann.** "Convolutional neural networks for the classification of guitar effects and extraction of the parameter settings of single and multi-guitar effects from instrument mixes," EURASIP JASMP 2022, DOI 10.1186/s13636-022-00257-4; "Blind extraction of guitar effects through blind system inversion and neural guitar effect modeling," EURASIP JASMP 2024, DOI 10.1186/s13636-024-00330-0. `[Crossref]` `[abstract]`. The 2022 paper reports up to 97.4% effect classification accuracy from mixes of guitar, bass, keyboard and drums; the 2024 paper reports a listening test with eight subjects. These are the Hinrichs et al. papers I could match to this topic; no mastering-specific Hinrichs paper was found.

Also: Yang, Berg-Kirkpatrick, McAuley, Novack, "WildFX," arXiv:2507.10534 (DAW-driven effect-graph data generation) `[arXiv]`.

### A5. Restoration and enhancement of full mixes

**Li, Luo (2025). "Apollo: Band-sequence Modeling for High-Quality Audio Restoration." ICASSP 2025.** DOI 10.1109/ICASSP49660.2025.10890825, arXiv:2409.08514. `[Crossref]` `[arXiv]` `[full text]`.
- Method: band-split front end with alternating band and sequence modelling (Roformer and TCN) to restore MP3-compressed music at 44.1 kHz.
- Data: MUSDB18-HQ plus MoisesDB stems, randomly remixed (1 to 8 stems, 3-second clips, gains within 10 dB), MP3-encoded at 24 to 128 kbit/s.
- Evaluation: SDR, SI-SNR, ViSQOL; 5,000 test samples per condition; baseline SR-GAN. No listening test.
- Code: https://github.com/JusperLee/Apollo (CC BY-SA 4.0); checkpoint `JusperLee/Apollo` on Hugging Face, ungated.

**Moliner, Elvander, Valimaki (2024). "Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach" (BABE). IEEE/ACM TASLP 32, 5092-5105.** DOI 10.1109/TASLP.2024.3507566, arXiv:2306.01433. `[Crossref]` `[arXiv]`.
- Method: diffusion posterior sampling with an unknown, parametrised lowpass degradation estimated during sampling. Data include MAESTRO at 22.05 kHz. Evaluation: log-spectral distance, Frechet distance, listening tests. Code: https://github.com/eloimoliner/BABE (MIT).

**Moliner, Turunen, Elvander, Valimaki (2024). "A Diffusion-Based Generative Equalizer for Music Restoration" (BABE-2). DAFx 2024.** arXiv:2403.18636. `[arXiv]` `[full text]`.
- Method: extends BABE from bandwidth extension to "generative equalization": jointly estimates a filter magnitude response and regenerates missing content, with an LTAS-based initialisation. Evaluation: FAD with several embeddings via fadtk, LTAS distance, historical piano and singing recordings. Code: https://github.com/eloimoliner/BABE2-music-restoration (MIT); checkpoints at research.spa.aalto.fi (reachable, no login). Earlier: BEHM-GAN, TASLP 31, 943-956, DOI 10.1109/TASLP.2022.3190726.

**Kandpal, Nieto, Jin (2022). "Music Enhancement via Image Translation and Vocoding." ICASSP 2022, pp. 3124-3128.** DOI 10.1109/ICASSP43922.2022.9747454, arXiv:2204.13289. `[Crossref]` `[full text]`.
- Method: mel-spectrogram image translation followed by a diffusion vocoder, for low-quality recordings of solo instruments (Medley-solos-DB, 16 kHz).
- Evaluation: MOS on Amazon Mechanical Turk, 211 listeners, 9,095 answers; Spearman correlation of objective metrics against MOS, with FAD the best aligned.

**Zang, Dai, Plumbley, Kong (2025). "Music Source Restoration."** arXiv:2505.21827. **MSRBench**, arXiv:2510.10995. **Challenge summary**, arXiv:2601.04343. `[arXiv]` `[abstract]`. Recover undegraded stems from mixtures whose stems went through EQ, compression, distortion, reverb and codecs. RawStems: 578 songs, 354.13 h. Code and models released (Hugging Face `yongyizang/MSR_UFormers`, Apache-2.0, ungated).

2026 restoration work, `[arXiv]` `[abstract]` unless noted:
- Svento, Moliner, Kallinen, Juvela, Valimaki, Rajmic. "Music Restoration via Latent Operator Optimization and Diffusion Model Priors" (LOUDAR). ISMIR 2026. arXiv:2608.01972. Unknown distortion modelled as a learnable latent operator; evaluated on singing-voice effect removal and guitar distortion.
- Cho, Koo, Lafargue, Dhyani, Moliner, Mitsufuji. "End-to-End Historical Music Restoration in Latent Space." arXiv:2610.00607 (submitted to ICASSP 2027). Latent flow matching on synthetic gramophone degradations; Audiobox PQ among metrics; 38 listeners `[full text]`.
- Karystinaios, Greif, Nadrchal, Primus, Widmer. "Low-Latency Neural Models for Real-Time Music Enhancement." arXiv:2607.12872. Causal models at 44.1 kHz evaluated on a music-and-noise set and the SonicMaster set with FAD-CLAP, KL, SI-SNR, SSIM, PQ and Zimtohrli; reports that metric families disagree and that indiscriminate enhancement can make inputs worse. Code: https://github.com/manoskary/audio-enhancement.
- Liu, Ai, Ling. "Neural Music Enhancement with Dual Time-Frequency Spectral Representations for Prediction and Discrimination" (DSME). ISCSLP 2026. arXiv:2609.03357.
- Review: Lemercier, Richter, Welker, Moliner, Valimaki, Gerkmann, "Diffusion Models for Audio Restoration: A review," IEEE SPM 41(6), 72-84, DOI 10.1109/MSP.2024.3445871 `[Crossref]`.

### A6. Stem-aware and separation-assisted remixing

**Yang, Firodiya, Bryan, Kim (2022). "Don't Separate, Learn to Remix: End-to-End Neural Remixing with Joint Optimization." ICASSP 2022, pp. 116-120.** DOI 10.1109/ICASSP43922.2022.9746077, arXiv:2107.13634. `[Crossref]` `[arXiv]` `[full text]`.
- Method: Conv-TasNet repurposed to output a remix directly given per-source gain controls, trained with a remix loss jointly with a separation loss.
- Data: Slakh (about 48 h training) and MUSDB18 (about 6 h), cross-dataset tests.
- Evaluation: minimum of SDR and SD-SDR on the remix, loudness difference per source. No listening test found in the text.
- Code: none found `[UNVERIFIED]`.

**Wierstorf, Ward, Mason, Grais, Hummersone, Plumbley (2017). "Perceptual Evaluation of Source Separation for Remixing Music." AES 143rd Convention, New York.** AES E-Library 19277. Stimuli: Zenodo DOI 10.5281/zenodo.835182 (authors Wierstorf and Ward) `[OpenAlex]`. Full author list and paper number `[UNVERIFIED]`.
- Method `[snippet]`: five separation algorithms used to raise the vocal level in six songs; listening test on loudness balance and sound quality.
- Headline `[snippet]`: some algorithms allowed vocal boosts up to 6 dB at the cost of perceptible quality loss.

**Roma, Grais, Simpson, Plumbley (2016). "Music Remixing and Upmixing Using Source Separation."** Workshop paper, University of Surrey repository. `[OpenAlex]`. **Simpson, Roma, Plumbley (2015). "Deep Remix: Remixing Musical Mixtures Using a Convolutional Deep Neural Network."** arXiv:1505.00289 `[arXiv]`. **Mimilakis, Cano, Abesser, Schuller (2016). "New Sonorities for Jazz Recordings: Separation and Mixing using Deep Neural Networks."** 2nd Workshop on Intelligent Music Production `[snippet]`: separates solo and accompaniment in old jazz recordings and remixes them, the closest academic precedent for separation-based remastering of archival mixes.

**Pons, Janer, Rode, Nogueira (2016). "Remixing music using source separation algorithms to improve the musical experience of cochlear implant users." JASA 140(6), 4338-4349.** DOI 10.1121/1.4971424. `[Crossref]`.

**Cadenza challenges.** Roa Dabike et al., "The first Cadenza challenges," arXiv:2409.05095; ICASSP 2024 challenge paper DOI 10.1109/ICASSPW62465.2024.10626340; Bannister et al., "The First Cadenza Challenge: Perceptual Evaluation...," Trends in Hearing 30, 2026, DOI 10.1177/23312165251408761. `[arXiv]` `[Crossref]`. Demix then remix pop/rock for a personalised rebalance; baselines Hybrid Demucs and Open-Unmix; objective metric HAAQI; 53 hearing-aid users rated eight systems and no entrant beat the baseline on audio quality `[snippet]`.

**Yeh, Koo, Martinez-Ramirez, Liao, Yang, Mitsufuji (2025). "Fx-Encoder++: Extracting Instrument-Wise Audio Effects Representations from Mixtures." ISMIR 2025.** arXiv:2507.02273. `[arXiv]` `[abstract]`. Contrastive encoder with an extractor that turns a mixture-level effects embedding into per-instrument embeddings given an audio or text query. Code: https://github.com/SonyResearch/Fx-Encoder_PlusPlus.

**Cheng, Wu, Chen, Yeh, Chen, Yang (2026). "StemFX: Learning Mixing Style Representations via Autoregressive FX Chain Prediction on Source-Separated Stems." ISMIR 2026.** arXiv:2607.15634. `[arXiv]` `[full text]`.
- Method: separates about 105K songs into pseudo-stems, augments them with MultiAFx (85 effects from 7 libraries), and trains a transformer to predict variable-length effect chains per stem.
- Evaluation: mixing style retrieval, paired style transfer with MR-STFT, MUSHRA-style test with 20 listeners with production experience; StemFX scored 60.6 and the hidden target 96.6.
- Code: https://github.com/barry-mir/stemfx and https://github.com/barry-mir/multiafx; `stemfx` 0.2.0 on PyPI (MIT).

Upmixing: Guo et al., "SPHERE: Automatic Music Upmixing via Audio Language Model Post-Training with Spatial Heuristic Rewards," EMNLP 2026, arXiv:2608.30559 (predicts spatial mixing parameters from stems) `[arXiv]`. Overview of separation: Cano, FitzGerald, Liutkus, Plumbley, Stoter, "Musical Source Separation: An Introduction," IEEE SPM 36(1), 31-40, DOI 10.1109/MSP.2018.2874719 `[Crossref]`.

### A7. Loudness, dynamics and spectral analyses; perceptual studies

- **Deruty, Tardieu (2014). "About Dynamic Processing in Mainstream Music." JAES 62(1/2), 42-55.** DOI 10.17743/jaes.2014.0001. `[Crossref]` `[abstract]`. Signal features on tracks from 1967 to 2011; the loudness war "may have peaked in 2004", reduced peak salience, and did not reduce long-term dynamics.
- **Deruty, Pachet (2015). "The MIR Perspective on the Evolution of Dynamics in Mainstream Music." ISMIR 2015.** `[OpenAlex]`.
- **Vickers (2010). "The Loudness War: Background, Speculation, and Recommendations." AES 129th Convention.** `[OpenAlex]`; paper number `[UNVERIFIED]`.
- **Serra, Corral, Boguna, Haro, Arcos (2012). "Measuring the Evolution of Contemporary Western Popular Music." Scientific Reports 2, 521.** DOI 10.1038/srep00521. `[Crossref]`.
- **Hove, Vuust, Stupacher (2019). "Increased levels of bass in popular music recordings 1955-2016 and their relation to loudness." JASA 145(4), 2247-2253.** DOI 10.1121/1.5097587. `[OpenAlex]`.
- **Pestana, Ma, Reiss, Barbosa, Black (2013). "Spectral Characteristics of Popular Commercial Recordings 1950-2010." AES 135th Convention, New York, Paper 8960.** AES E-Library 17010. `[OpenAlex]` `[snippet]`. Long-term average spectra of a large corpus by year and genre; finds a consistent tendency toward a target equalization curve. This is the standard citation for target-curve mastering EQ.
- **Hjortkjaer, Walther-Hansen (2014). "Perceptual Effects of Dynamic Range Compression in Popular Music Recordings." JAES 62(1/2), 37-41.** DOI 10.17743/jaes.2014.0003. `[Crossref]` `[abstract]`. Originals vs more compressed remasters; no evidence that compression affected preference or perceived depth.
- **Croghan, Arehart, Kates (2012). "Quality and loudness judgments for music subjected to compression limiting." JASA 132(2), 1177-1188.** DOI 10.1121/1.4730881. `[Crossref]` `[abstract]`. Scaled paired comparison; light compression preferred when loudness varied, heavy compression harmful with or without loudness equalization.
- **Ronan, Ward, Sazdov, Lee (2017). "The Perception of Hyper-Compression by Mastering Engineers." JAES 65(7/8), 613-621.** DOI 10.17743/jaes.2017.0023. `[OpenAlex]` `[abstract]`. ABX with 20 mastering engineers; 17 of 24 conditions discriminated; audibility tied to crest factor.
- **Wilson, Fazenda (2016). "Perception of Audio Quality in Productions of Popular Music." JAES 64(1/2), 23-34.** DOI 10.17743/jaes.2015.0090. `[Crossref]` `[snippet]`. Commercial CD excerpts rated for quality and liking; quality ratings associated most with features tied to loudness and dynamic range compression, liking with familiarity. Listener and excerpt counts `[UNVERIFIED]`.
- **Wilson, Fazenda (2016). "Variation in Multitrack Mixes: Analysis of Low-level Audio Signal Features." JAES 64(7/8), 466-473.** DOI 10.17743/jaes.2016.0029. `[Crossref]` `[snippet]`. 1,501 mixes of 10 songs; principal dimensions of variation were amplitude, brightness, bass and width.
- **Wilson, Fazenda (2015). "101 Mixes: A Statistical Analysis of Mix-Variation in a Dataset of Multi-Track Music Mixes." AES 139th Convention.** `[OpenAlex]`.
- **De Man, Reiss (2017). "The Mix Evaluation Dataset." DAFx-17.** 180 mixes with DAW sessions and close to 5,000 preference ratings (verified from the DAFx PDF).

### A8. Text-controlled mixing and mastering (2022 to 2026)

- **Venkatesh, Moffat, Miranda (2022). "Word Embeddings for Automatic Equalization in Audio Mixing." JAES 70(9), 753-763.** DOI 10.17743/jaes.2022.0047, arXiv:2202.08898. `[Crossref]`.
- **Zheng, Seetharaman, Pardo (2016). "SocialFX: Studying a Crowdsourced Folksonomy of Audio Effects Terms." ACM Multimedia 2016, pp. 182-186.** DOI 10.1145/2964284.2967207. `[Crossref]`. Source of the descriptor data reused by LLM2Fx.
- **Chu, O'Reilly, Barnett, Pardo (2025). "Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects." ICASSP 2025.** DOI 10.1109/ICASSP49660.2025.10890334, arXiv:2409.18847. `[Crossref]` `[full text]`. Single-instance optimisation of differentiable EQ and reverb parameters against a CLAP text target; crowd listener study with 167 participants after screening. Code: https://github.com/anniejchu/text2fx (no license file). SonicMaster uses it as its EQ baseline.
- **Doh, Koo, Martinez-Ramirez, Liao, Nam, Mitsufuji (2025). "Can Large Language Models Predict Audio Effects Parameters from Natural Language?" (LLM2Fx). WASPAA 2025.** DOI 10.1109/WASPAA66052.2025.11230953, arXiv:2505.20770. `[Crossref]` `[full text]`. Zero-shot and few-shot prediction of EQ and reverb parameters from text with GPT-4o, Llama 3 variants and Mistral-7B, evaluated on a cleaned SocialFX set with an MMD measure; reports GPT-4o EQ MMD 0.22 and Llama3.3-70B 0.24.
- **Doh et al. (2026). "LLM2Fx-Tools: Tool Calling For Music Post-Production." ICLR 2026.** arXiv:2512.01559. `[arXiv]` `[full text]`. Qwen3-4B with an audio adapter emits an executable effect chain with chain-of-thought; LP-Fx dataset built from MedleyDB raw tracks; reports 80% effect-type accuracy and 0.56 Spearman correlation for ordering; LLM-as-judge with GPT-5.
- **Clemens, Marasovic (2025). "MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing." COLM 2025.** arXiv:2507.06329. `[arXiv]` `[abstract]`. 431 audio-grounded dialogue turns from 7 sessions with 12 producers; fine-tuned Qwen-Audio ranks best.
- **Schaffer, Singh (2026). "RIME: Enabling Large-Scale Agentic Music Post-Production."** arXiv:2607.19605. `[arXiv]` `[full text]`. Rule-based generator of paired edit instructions and effect graphs; 10 listeners, 900 ratings; FAD and KAD; code at github.com/sahaslab/RIME.
- **Yu et al. (2026). "InstructFX2FX: A Multi-Turn Text-to-Effect System for Sequential Audio Effect Refinement." DAFx26, pp. 526-529.** arXiv:2606.22005. `[arXiv]` `[abstract]`. LLM planner plus CLAP-guided refinement over turns.
- SonicMaster (A1) and the CLAP-text mode of ITO-Master (A1) are the two text-controlled systems that operate on full stereo mixes for mastering.

---

## B. Runnable automated mastering baselines

Checks were made on 2026-10-02 with the GitHub API, PyPI, Hugging Face API, Zenodo API and HTTP HEAD requests. "Open weights" means a download succeeded or returned HTTP 200 without any account, token or click-through.

### B0. Summary

| Baseline | Task fit for stereo-mix mastering | Reference track needed | Weights open without login | License | Verdict |
|---|---|---|---|---|---|
| Matchering 2.0 | Direct | Yes | No weights (DSP) | GPL-3.0 | Runs today via pip |
| ffmpeg `loudnorm` / ffmpeg-normalize | Loudness only | No | No weights | LGPL/GPL (ffmpeg); MIT-style (ffmpeg-normalize) | Runs today; lower bound |
| pedalboard chain + pyloudnorm | Fixed chain | No | No weights | GPL-3.0 (pedalboard), MIT (pyloudnorm) | Runs today; hand-built baseline |
| ITO-Master | Direct | Yes (audio), or text via CLAP ITO | Yes (public HF Space) | CC BY-NC 4.0 | Runs via pip plus repo script |
| E2E Remastering (Koo 2022) | Direct | Yes | Yes (public Google Drive) | MIT | Runs; older PyTorch stack |
| DeepAFx-ST | EQ plus compressor style transfer | Yes | Yes (GitHub release, 1.09 GB) | Adobe Research License, noncommercial | Runs; models trained on 24 kHz mono |
| SonicMaster | Direct (restoration plus enhancement) | No (text prompt or auto) | Model weights yes; required VAE is gated | Apache-2.0 (code and weights) | Blocked without a Hugging Face login for Stable Audio Open 1.0 |
| Music Mixing Style Transfer (Koo 2023) | Per-stem style transfer on separated mix | Yes | Yes (public Google Drive) | MIT | Runs; uses Demucs internally |
| FxNorm-Automix | Mixing from 4 stems | No | Yes (in repository) | MIT | Needs stems and user-supplied impulse responses |
| Diff-MST | Multitrack mixing | Yes | No checkpoint published | CC BY-NC-SA 4.0 | Not runnable without training |
| phaselimiter (AI Mastering engine) | Direct | No | Prebuilt binaries in GitHub releases | MIT | Runnable; undocumented CLI |
| master_me | Live-stream leveling plugin | No | No weights | GPL-3.0 | No CLI; README says it is not meant for mastering recorded music |

### B1. Matchering 2.0

- URL: https://github.com/sergree/matchering (2,651 stars, last push 2026-07-08). PyPI `matchering` 2.0.6, Python >= 3.8.
- License: GPL-3.0.
- What it does: matches RMS, frequency response, peak amplitude and stereo width of a target to a reference, with its own limiter. No learned weights.
- Input/output: WAV or anything libsndfile reads (MP3 with ffmpeg installed); writes 16-bit or 24-bit PCM WAV.
- Reference: required.
- Install: `python3 -m pip install -U matchering` (Linux also needs `libsndfile1`).
- Minimal use (from the README):

```python
import matchering as mg

mg.process(
    target="my_song.wav",
    reference="some_popular_song.wav",
    results=[
        mg.pcm16("my_song_master_16bit.wav"),
        mg.pcm24("my_song_master_24bit.wav"),
    ],
)
```

- CLI alternative: https://github.com/sergree/matchering-cli (GPL-3.0): `python3 mg_cli.py my_song.wav some_popular_song.wav my_song_master_16bit.wav`.
- Used as a baseline in ITO-Master (Table 1). Songmastr (songmastr.com) is a hosted front end for it.

### B2. ffmpeg `loudnorm`, ffmpeg-normalize, pyloudnorm

- ffmpeg `loudnorm` implements EBU R128 normalization with integrated loudness, loudness range and true-peak targets (options `I`, `LRA`, `TP`, `measured_*`, `linear`, `print_format`; confirmed against the local ffmpeg build and the ffmpeg-filters documentation). The documentation states that in dynamic mode the stream is upsampled to 192 kHz for true-peak detection and that `-ar` should be used to set the output rate.
- Single pass:

```bash
ffmpeg -i in.wav -af loudnorm=I=-14:TP=-1:LRA=11 -ar 44100 out.wav
```

- Two pass (measure, then apply linearly where possible):

```bash
ffmpeg -hide_banner -i in.wav -af loudnorm=I=-14:TP=-1:LRA=11:print_format=json -f null -
ffmpeg -i in.wav -af loudnorm=I=-14:TP=-1:LRA=11:measured_I=<input_i>:measured_TP=<input_tp>:measured_LRA=<input_lra>:measured_thresh=<input_thresh>:offset=<target_offset>:linear=true -ar 44100 out.wav
```

- ffmpeg-normalize (https://github.com/slhck/ffmpeg-normalize, 1,538 stars, PyPI 1.42.0, Python >= 3.10) wraps the two-pass flow: `pip3 install ffmpeg-normalize` then `ffmpeg-normalize in.wav -t -14 -o out.wav` (`-t` target level; a `--true-peak` option exists in the CLI source).
- pyloudnorm (https://github.com/csteinmetz1/pyloudnorm, MIT, PyPI 0.2.0) gives BS.1770 measurement and static gain normalization in Python:

```python
import soundfile as sf
import pyloudnorm as pyln

data, rate = sf.read("in.wav")
meter = pyln.Meter(rate)
loudness = meter.integrated_loudness(data)
out = pyln.normalize.loudness(data, loudness, -14.0)
sf.write("out.wav", out, rate)
```

- Role: a loudness-only lower bound. No tonal or dynamic shaping beyond the dynamic mode of `loudnorm`. No reference needed.

### B3. Pedalboard-based chain

- URL: https://github.com/spotify/pedalboard (6,332 stars, GPL-3.0, PyPI 0.9.25, Python >= 3.10).
- A GitHub sweep found no widely used standalone pedalboard mastering script; hits were ComfyUI nodes and small personal repositories. The snippet below is therefore my own minimal chain, built only from classes named in the pedalboard README (`HighpassFilter`, `Compressor`, `Gain`, `Limiter`, `AudioFile`). It is a fixed "naive engineer" baseline and has no published pedigree.

```python
from pedalboard import Pedalboard, HighpassFilter, Compressor, Gain, Limiter
from pedalboard.io import AudioFile

board = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=30),
    Compressor(threshold_db=-18, ratio=2, attack_ms=30, release_ms=200),
    Gain(gain_db=4),
    Limiter(threshold_db=-1.0, release_ms=100),
])

with AudioFile("in.wav") as f:
    audio, sr = f.read(f.frames), f.samplerate
out = board(audio, sr)
with AudioFile("out.wav", "w", sr, out.shape[0]) as f:
    f.write(out)
```

- Follow with the pyloudnorm snippet in B2 to hit a loudness target. Note the GPL-3.0 license if code is redistributed. The snippet was not executed during this survey (pedalboard and matchering are not installed in the local environment; pyloudnorm, torch 2.8 and ffmpeg are).

### B4. ITO-Master (Sony)

- URL: https://github.com/SonyResearch/ITO-Master (29 stars). PyPI `ito-master` 0.1.1.
- License: CC BY-NC 4.0 (LICENSE file and PyPI metadata).
- Weights: `inference.py` calls `hf_hub_download` on the public Hugging Face Space `jhtonyKoo/ITO-Master` (`models/white_box_converter.pt`, 42 MB, HTTP 200 without a token; encoder and black-box weights in the same folder). No gating.
- Input/output: audio files (examples are FLAC), default 44.1 kHz; writes to an output directory; output loudness normalization on by default.
- Reference: required for the base style transfer. The ITO step can target a text prompt through CLAP instead of the audio reference.
- Install (from the README): `sudo apt-get install -y libsox-fmt-all libsox-dev sox libsndfile1` then `pip install ito_master`; clone the repository for `inference.py` and `fxnorm_feat.npy`.
- Minimal command (from the README; use `--inference_device cpu` when no GPU is present):

```bash
python inference.py \
  --input_path examples/input_1.flac \
  --reference_path examples/reference_1.flac \
  --model_type white_box \
  --inference_device cuda \
  --output_dir_path outputs/white_box_st/
```

- With inference-time optimization add `--perform_ito --ito_reference_path <ref> --ito_objective AudioFeatureLoss --num_steps 100`. Text target: `--ito_objective CLAPFeatureLoss --clap_target_type Text --clap_text_prompt "heavy metal"`.
- The apt lines assume Linux; on macOS the sox dependency needs a Homebrew equivalent `[UNVERIFIED]`.

### B5. End-to-end Music Remastering System (Koo, Paik, Lee 2022)

- URL: https://github.com/jhtonyKoo/e2e_music_remastering_system (47 stars, MIT).
- Weights: two Google Drive files (Music Effects Encoder, Mastering Cloner). Both links answered with the public large-file interstitial, so `gdown` works without a Google login.
- Input/output: stereo 44.1 kHz 16-bit WAV. Files must be named `input.wav` and `reference.wav` inside one folder per song.
- Reference: required.
- Minimal command (from the README):

```bash
python inference.py \
    --ckpt_dir "path_to_checkpoint_directory" \
    --data_dir_test "path_to_directory_containing_inference_samples"
```

- This is the "E2E Remastering" baseline in the ITO-Master paper.

### B6. DeepAFx-ST (Adobe Research)

- URL: https://github.com/adobe-research/DeepAFx-ST (413 stars, last push 2023-05-30).
- License: Adobe Research License, noncommercial research use only, for code and models.
- Weights: GitHub release asset `checkpoints_and_examples.tar.gz` (1,092,031,781 bytes, HTTP 200, no login).
- Input/output: WAV in, WAV out. The style encoders were trained at 24 kHz mono; the DSP effects (parametric EQ and compressor) run at the input rate. Music checkpoints are the `jamendo` ones.
- Reference: required.
- Install (from the README): Python 3.8 environment, `pip install --pre -e .`, plus `libsndfile1`, `sox`, `ffmpeg`.
- Minimal commands (from the README):

```bash
wget https://github.com/adobe-research/DeepAFx-ST/releases/download/v0.1.0/checkpoints_and_examples.tar.gz -O - | tar -xz
python scripts/process.py -i <input_audio>.wav -r <ref_audio>.wav \
  -c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt
```

### B7. SonicMaster (AMAAI Lab)

- URL: https://github.com/AMAAI-Lab/SonicMaster (201 stars, last push 2026-09-16).
- License: Apache-2.0 (repository and Hugging Face model card).
- Weights: `amaai-lab/SonicMaster` on Hugging Face, `gated: false`, `model.safetensors` 3.45 GB.
- Gating problem: both inference scripts call `AutoencoderOobleck.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae")`. That repository reports `gated: auto` and returned HTTP 401 for `vae/config.json` without a token. Running SonicMaster as released therefore needs a Hugging Face account that has accepted the Stable Audio Open license. This fails the "no login" criterion.
- Input/output: any audio file readable by the script, processed at 44.1 kHz stereo in 30-second chunks with 10-second overlap; writes WAV or FLAC.
- Reference: none. A text prompt is required by `infer_single.py`.
- Install: the README says Python 3.13 and `pip install -r requirements_sonic.txt` (pins torch 2.4.0, diffusers 0.30.0, transformers 4.44.0, laion_clap 1.1.7). The README has no inference section; the command below comes from the argparse block of `infer_single.py`.

```bash
python infer_single.py \
  --ckpt /path/to/model.safetensors \
  --input in.wav \
  --prompt "Master this track for me, please!" \
  --output out.wav
```

- The prompt string above is one of the generic phrases the paper lists for its automatic mode. `inference_fullsong.py` handles batch full-song inference from a JSONL manifest.
- A hosted demo exists at https://huggingface.co/spaces/amaai-lab/SonicMaster.

### B8. FxNorm-Automix (Sony)

- URL: https://github.com/sony/FxNorm-automix (148 stars, MIT).
- Weights: stored in the repository under `trainings/results/` (`ours_S_Lb`, `ours_S_pretrained`, `wun_S_Lb`, about 10 to 11 MB each) with `trainings/features/features_MUSDB18.npy`. The README says "training/results"; the actual folder is `trainings/results`.
- Input: four stems (vocals, bass, drums, other). For a stereo mix a separator must run first.
- Impulse responses: the IRs used in the paper are not public; the loader expects user-supplied folders, each with `impulse_response.wav`.
- Reference: none.
- Install: `python setup.py install` and `pip install -r requirements.txt` (pins torch 1.9.0).
- Minimal command (from `scripts/inference.sh`):

```bash
python automix/inference.py --vocals wet_vocals.wav --bass wet_bass.wav \
  --drums wet_drums.wav --other wet_other.wav \
  --output mixes/mix_from_wet_stems.wav \
  --training-params configs/ISMIR/ours_S_Lb.py \
  --impulse-responses /path/to/IR \
  --nets trainings/results/ours_S_Lb/net_mixture.dump \
  --weights trainings/results/ours_S_Lb/current_model_for_mixture.params \
  --features trainings/features/features_MUSDB18.npy
```

- The script names `best_model_for_mixture_valid_stereo_loss_mean.params`; in the repository tree `ours_S_Lb` contains `current_model_for_mixture.params`, so the weights path above follows the tree.

### B9. Music Mixing Style Transfer (FXencoder plus MixFXcloner)

- URL: https://github.com/jhtonyKoo/music_mixing_style_transfer (182 stars, MIT). Hosted demo on Hugging Face Spaces.
- Weights: two public Google Drive files (FXencoder, MixFXcloner), trained on MUSDB18.
- Input/output: stereo 44.1 kHz 16-bit WAV, one folder per song with input and reference files.
- Reference: required.
- Behaviour on mixes: by default the script separates both input and reference with Demucs (`--separation_model mdx_extra`), converts each of drums, bass, other and vocals, and sums them. `--do_not_separate` disables this.
- Minimal command (from the README):

```bash
python inference/style_transfer.py \
    --ckpt_path_enc "path_to_checkpoint_of_FXencoder" \
    --ckpt_path_conv "path_to_checkpoint_of_MixFXcloner" \
    --target_dir "path_to_directory_containing_inference_samples"
```

### B10. Diff-MST

- URL: https://github.com/sai-soum/Diff-MST (65 stars), CC BY-NC-SA 4.0.
- Weights: none in the repository or its releases; issue #27 requesting checkpoints is open. The evaluation scripts point to local paths on the authors' machines.
- Input: raw multitracks plus a reference mix. Not applicable to a single stereo mix without separation.
- Status: usable only after training (`python main.py fit -c configs/config.yaml -c configs/optimizer.yaml -c configs/data/medley+cambridge-8.yaml -c configs/models/naive.yaml`).

### B11. phaselimiter (engine behind aimastering.com)

- URL: https://github.com/ai-mastering/phaselimiter (51 stars), README states MIT and project status "inactive". GUI: https://github.com/ai-mastering/phaselimiter-gui (118 stars, Windows binaries).
- Binaries: releases v0.2.0 (2023-08-21, `release.tar.xz`, `phaselimiter-win.zip`) and v0.1.0 (with `install_linux.sh`). No macOS binary.
- Reference: none; a target loudness is passed.
- Command: the README has no usage section. The GUI source calls `phase_limiter --input <in> --output <out> --ffmpeg <ffmpeg> --mastering true --mastering_mode mastering5 --reference <loudness>`. Treat this invocation as `[UNVERIFIED]` until tested.

### B12. Other open-source chains found on GitHub

| Repo | Stars | License | Notes |
|---|---|---|---|
| trummerschlunk/master_me | 751 | GPL-3.0 | Faust plugin (CLAP, VST, LV2, JACK) for live streaming. README: "it is NOT intended to automatically master your recorded music." No CLI. |
| eas4ai/Web-Audio-Mastering | 120 | ISC | Browser GUI: -14 LUFS normalization, true-peak limiter, EQ, glue compression. No CLI. |
| Wamphyre/oXygen | 45 | BSD-3-Clause | JUCE VST3 with deterministic assistant and reference match. No CLI. |
| fadelabs/phantom | 37 | AGPL-3.0 | Python CLI that wraps Pedalboard and Matchering: `phantom render mix.wav --reference reference.wav --output matched.wav`. |
| libraz/libsonare | 26 | Apache-2.0 | C++ library with Python binding: `audio.mastering(target_lufs=-14.0, ceiling_db=-1.0)`. |
| ai-mastering/aimastering-tools | 34 | MIT | Client for the hosted AI Mastering API; processing happens on the vendor's servers. |

Star counts and commands in this table come from a GitHub sweep performed by a helper agent with the `gh` CLI; the master_me quotation was re-read from the README.

### B13. Restoration-side baselines (same input type, different task)

- **Apollo** (https://github.com/JusperLee/Apollo, 422 stars, CC BY-SA 4.0). Restores lossy-codec damage at 44.1 kHz. Checkpoint `JusperLee/Apollo` on Hugging Face, ungated, downloaded automatically. `python inference.py --in_wav=input.wav --out_wav=output.wav --device=auto`.
- **De-limiter** (https://github.com/jeonchangbin49/De-limiter, 96 stars, MIT). Undoes limiting. Weights (9 MB) in `./weight`. Put files in `./input_data` or pass `--data_root=/path/to/music.wav`, then `python -m inference` (output directory flag `--output_directory`).
- **RemFX** (https://github.com/mhrice/RemFx, Apache-2.0). `scripts/download_ckpts.sh` then `scripts/remfx_detect.sh example.wav -o dry.wav`. Checkpoints on Zenodo 8218621 (open). Trained on single-source 48 kHz clips, so full mixes are out of distribution.
- **BABE-2** (https://github.com/eloimoliner/BABE2-music-restoration, MIT). Checkpoints at http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/. `python test.py --config-name=conf_singing_voice.yaml tester=singer_evaluator_BABE2 tester.checkpoint="path/to/checkpoint.pt" id="BABE2_restored" tester.evaluation.single_recording="path/to/recording.wav"`. Models cover piano and singing voice.
- **MSR U-Formers** (Hugging Face `yongyizang/MSR_UFormers`, Apache-2.0, ungated). Per-stem restoration from degraded mixtures.
- **Text2FX** (https://github.com/anniejchu/text2fx, no license file). `python -m text2fx.apply assets/guitar.wav eq 'warm like a hug' --export_dir experiments/ --learning_rate 0.01 --params_init_type random --n_iters 600 --criterion cosine-sim`. Text-driven EQ and reverb; SonicMaster's EQ baseline.
- **ST-ITO** (https://github.com/csteinmetz1/st-ito, Apache-2.0). `python scripts/run_optim.py "input.wav" --target "target.wav" --algorithm es --effect-type vst --dropout 0.0 --max-iters 25 --metric param`. Reference-based; searches plugin parameters.
- **automix-toolkit** (DMC) and **MEGAMI** have public checkpoints but take multitracks.

### B14. Commercial services (public information only)

Vendor pages were fetched by a helper agent on 2026-10-02; items marked `[UNVERIFIED]` there are repeated here.

| Service | Since | What the vendor or a primary source says | Reference upload | API |
|---|---|---|---|---|
| LANDR | 2014 | "Uses AI to pull knowledge from thousands of mastered songs to offer a unique mastering chain for every track"; three intensities and styles in the API. Patent US9304988B2 (Terrell, Mansbridge, Reiss, De Man; LANDR Audio; granted 2016) claims production driven by semantic rules. Sterne and Razlogova note the actual role of machine learning is unknowable from outside. | Yes | Yes (landr.com/pro-audio-mastering-api) |
| eMastered | 2016 | "Made by Grammy-winning engineers, powered by AI". Algorithm undisclosed. | `[UNVERIFIED]` | None found |
| CloudBounce | 2016 | Described in press as algorithms plus "AI features" applying EQ, compression and limiting. The domain did not resolve on 2026-10-02; service status `[UNVERIFIED]`. | n/a | n/a |
| BandLab Mastering | 2016 | Preset-based (8 presets designed with named mastering engineers), intensity control and reference matching on paid tiers. Algorithm undisclosed. | Paid tiers | None found |
| iZotope Ozone Master Assistant | 2017 (Ozone 8) | The Ozone 8 manual documents a "neural net classifier" that maps audio to a target curve mixed from 10 genre classes, then sets EQ, dynamics, Maximizer and dynamic EQ toward a Streaming (-14 LUFS, -1 dB ceiling), CD, or Reference target. Ozone 9 added Master Rebalance; Ozone 11 added Stem Focus and Clarity; Ozone 12 added Unlimiter and Stem EQ (Advanced edition). | Yes | No (plugin) |
| Waves Online Mastering | 2023 | "Waves Neural Networks technology, an AI mastering engine"; style and tone controls. | Yes | None found |
| Apple Logic Pro Mastering Assistant | 2023 (Logic 10.8) | Analyses the mix and adjusts dynamics, frequency balance, timbre and loudness; four characters; loudness knob centred near -14 LUFS-I; no reference input documented. | No | No |
| Masterchannel / TuneCore Mastering | 2022 / 2024 | Vendor says the system is "based on human feedback from engineers"; reinforcement-learning claims `[UNVERIFIED]`. | n/a | Yes (Masterchannel) `[UNVERIFIED]` |
| RoEx (Automix, Tonn API) | 2023 / 2025 | Built on Queen Mary research; mixing up to 32 tracks, mastering and mix analysis by API. Source of the Mourgela et al. dataset analysis. | n/a | Yes |
| AI Mastering (Bakuage) | n/a | Free hosted mastering with target loudness; engine open-sourced as phaselimiter (B11). | No | Yes |
| Slate Virtu, DistroKid Mixea, SoundCloud (Dolby), MasteringBOX, Maastr, sonible smart:limit | various | Style or intensity presets with loudness targets; algorithms undisclosed. | Varies | Dolby.io has a Music Mastering API |

No first-party Spotify mastering service was found.

### B15. Published comparisons involving commercial mastering services

1. Martinez Ramirez et al. 2021 (DeepAFx): LANDR as baseline in a 17-listener MUSHRA on mono 22 kHz 4-second clips; medians 0.88 (DeepAFx) vs 0.79 (LANDR), difference not significant at their threshold.
2. Elliott and Chon 2022 (JAES): two human engineers vs two automated services; listeners could not reliably pick the human masters; genre-dependent preference.
3. Piotrowska et al. 2017 (AES 142): signal analysis of eight "instant mastering" services, no listeners.
4. Mimilakis et al. 2016 (AES 140): abstract mentions improvement over "relevant and commercial software"; details not accessed.
5. Aker 2024, "AI-Assisted Music Mastering," IGI Global chapter, DOI 10.4018/979-8-3693-7235-7.ch002: compares AI platforms with professional engineers; protocol behind a paywall `[UNVERIFIED]`.
6. Non-academic but large: a blind study by Benn Jordan reported by MusicTech on 2024-10-29 with 472 participants on one track ranked two human engineers first and second and Matchering 2.0 third, ahead of several commercial tools. https://musictech.com/news/gear/benn-jordan-ai-mastering-study

No ML paper found uses eMastered, CloudBounce, BandLab, Waves, Logic or Ozone Master Assistant as a scored baseline. ITO-Master and SonicMaster compare only against open systems.

---

## C. Datasets usable for mastering research

Facts in this section were collected by a helper agent from Zenodo, Hugging Face and GitHub APIs, dataset pages and papers, then spot-checked (Hugging Face gating flags, Zenodo access rights and sizes, and the Cambridge-MT unmastered-mix count were re-queried and matched).

### C1. Unmastered vs mastered pairs

**Verdict: no public dataset of real mastering-engineer pairs at scale was found.** Searches of Hugging Face, Zenodo, GitHub and Kaggle for "unmastered", "premaster" and "mastering dataset" returned only the SonicMaster dataset.

| Source | What the pairs are | Size | License / access |
|---|---|---|---|
| Cambridge-MT "Mixing Secrets" library (https://cambridge-mt.com/ms/mtk/) | Real: an "Unmastered Mix (WAV)" is listed next to a "Full Preview (MP3)" for a subset of projects. The page does not state that the preview is a master of that exact mix. | 225 of 625 projects in the 2025-07-17 Wayback snapshot | Free "for educational purposes only"; the FAQ says research use has not been agreed with contributors. No login. The preview host blocks scripted access. Reference side is lossy MP3. |
| DeepAFx mastering set (Martinez Ramirez et al. 2021) | Real: 138 unmastered and mastered track pairs drawn from Mixing Secrets | 429.3 / 51.1 / 50.3 min | Not released |
| SonicMaster dataset (https://huggingface.co/datasets/amaai-lab/SonicMasterDataset) | Synthetic: clean Jamendo clips with degraded versions from 19 degradation functions plus text prompts | About 25k clean 30-s clips, 7 degraded variants each (175,000 rows, 208 h clean), FLAC 44.1 kHz, 505 GB | Card says CC BY 2.0; underlying tracks carry per-track CC licenses. Not gated. Sources are Jamendo MP3s, so the clean side is likely of lossy origin (inference). |
| musdb-L / musdb-XL (Zenodo 7041331) | Synthetic, limiter only: Ozone 9 Maximizer applied to MUSDB18-HQ test | 50 songs; shipped as sample-wise gain ratios, 9.35 GB | CC BY 4.0, open. Needs MUSDB18-HQ. |
| musdb-XL-train (Zenodo 12194067) | Synthetic, limiter only | 300,000 four-second segments plus 100 songs, 18.2 GB as gain ratios | CC BY 4.0, open |
| MSRBench (https://huggingface.co/datasets/yongyizang/MSRBench) | Real engineers processed and mastered mixtures, but the pairing is raw stem vs degraded mixture; no unmastered mix | 2,000 ten-second clips, 13 conditions, FLAC 48 kHz, 28.4 GB | CC BY-NC 4.0, not gated |
| Mix Evaluation Dataset (De Man and Reiss 2017) | Multiple human mixes per song with ratings; mixes only | 180 mixes, about 5,000 ratings | Hosted on the Open Multitrack Testbed, which returned 403 on 2026-10-02 |

ITO-Master, Koo et al. 2022, Diff-MST and Matchering released no pair data. Practical substitutes: (a) synthetic mastering chains applied to unmastered mixes from MoisesDB, MUSDB18-HQ, MedleyDB or Cambridge-MT; (b) unpaired learning from mastered corpora such as MTG-Jamendo and FMA, which are MP3.

### C2. Multitrack and stem datasets

| Dataset | Size | Format | License | Download / login |
|---|---|---|---|---|
| MUSDB18 (Zenodo 1117372) | 150 tracks (100/50), about 10 h, 4.68 GB | STEMS mp4, AAC 256 kbps, 44.1 kHz (lossy) | Noncommercial, educational; per-track source licenses | Zenodo reports open access |
| MUSDB18-HQ (Zenodo 3338373) | 150 tracks, 22.66 GB | Stereo WAV 44.1 kHz | Same as MUSDB18 (`other-nc`) | Direct, no login. Mixtures are stem sums without mastering. |
| DSD100 | 100 tracks, about 14 to 16 GB | Stereo WAV 44.1 kHz | Defers to Mixing Secrets terms | Direct, no login |
| MoisesDB (https://github.com/moises-ai/moises-db; arXiv:2307.15913) | 240 tracks, 14 h 24 min, 12 genres | 44.1 kHz | CC BY-NC-SA 4.0, noncommercial research | Download button on music.ai/research; a form is likely `[UNVERIFIED]`. The paper says no compression, EQ or effects in mixing and no mastering. |
| MedleyDB 1.0 / 2.0 (Zenodo 1649325, 1715175) | 122 + 74 = 196 multitracks with mix, stems, raw | WAV 44.1 kHz 16-bit | CC BY-NC-SA 4.0 | Zenodo records are restricted; access request required |
| Slakh2100 (Zenodo 4599666) | 2,100 tracks, 145 h, 104 GB | FLAC 44.1 kHz 16-bit, synthesized from MIDI | CC BY 4.0 | Direct, no login |
| Cambridge-MT / Mixing Secrets | 625 projects listed | ZIPs of WAV, 16 or 24-bit, 44.1 kHz | Educational use only | No login; site blocks scripted access |
| RawStems (https://huggingface.co/datasets/yongyizang/RawStems; arXiv:2505.21827) | 578 songs, 354.13 h, 8 instrument groups, 213.8 GB | Derived from Mixing Secrets | Audio under original terms; annotations Apache-2.0 | Not gated |
| Open Multitrack Testbed (De Man et al., AES 137 e-Brief 165, 2014) | Counts `[UNVERIFIED]` | Varies | Creative Commons per contribution | Site returned 403 on 2026-10-02 |
| AAM (Zenodo 5794629) | 3,000 synthetic tracks, 209.6 GB | FLAC 44.1 kHz | CC BY 4.0 | Open |
| Divide and Remaster (Zenodo 5574713) | 3,295 / 440 / 652 one-minute mixtures, 106 GB | WAV 44.1 kHz mono; speech, music, effects | CC BY 4.0 | Open. Cinematic audio, music from FMA. |
| Telefunken "Live from the Lab" | About 10 sessions | WAV 24-bit 48 kHz | Home studio and educational use | Signup `[UNVERIFIED]` |

### C3. Commercially mastered or released music that is legally downloadable

| Corpus | Size | Codec | License | Download / login |
|---|---|---|---|---|
| FMA (https://github.com/mdeff/fma; arXiv:1612.01840) | 106,574 tracks, 343 days; fma_small 8,000 clips (7.68 GB), fma_medium 25,000 (23.8 GB), fma_large 106,574 clips (100.3 GB), fma_full untrimmed (943.6 GB) | MP3, 44.1 kHz stereo, mostly 320 kbit/s nominal (lossy) | Per-track Creative Commons; metadata CC BY 4.0 | `curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip` etc., no login |
| MTG-Jamendo (https://github.com/MTG/mtg-jamendo-dataset) | 55,701 full tracks, 508 GB | MP3 320 kbps (lossy) | Per-track CC; noncommercial research use | `python3 scripts/download/download.py --dataset raw_30s --type audio <dir> --unpack`, no login |
| JamendoMaxCaps (https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps; arXiv:2502.07461) | 362,298 instrumental tracks, 1.06 TB | MP3 (lossy) | Card CC BY-SA 3.0; per-track licenses vary | Not gated |
| Song Describer Dataset (Zenodo 10072001; arXiv:2311.10057) | 706 tracks, about 1.1k captions, 3.32 GB | MP3 320 kbps 44.1 kHz (lossy) | Captions CC BY-SA 4.0; audio per-track CC | Open |
| MusicCaps | 5,521 ten-second clips | YouTube IDs only | CC BY-SA 4.0 (annotations) | No audio distributed |
| Internet Archive netlabels | 80,466 items | MP3 / OGG, per item | Often CC, per item | Open; no packaged dataset |
| MAESTRO v3 | 198.7 h piano | WAV 44.1 to 48 kHz | CC BY-NC-SA 4.0 | Open; piano only |

Unsuitable or pointer-only for full-band mastering work: AudioSet (IDs and features), Million Song Dataset (features), DISCO-10M (links and embeddings), Harmonix (spectrograms and YouTube URLs), MagnaTagATune (low-bitrate MP3), GTZAN (22.05 kHz mono, no stated license), MusicBench (includes 16 kHz material).

Every large mastered corpus surveyed here that can be downloaded legally is MP3. Lossless, mastered, redistributable music at scale was not found. Lossless audio exists only in the multitrack sets, whose mixtures are unmastered.

### C4. Genre-labelled corpora

| Corpus | Tracks | Genre labels |
|---|---|---|
| FMA | 106,574 | 161 genres in a hierarchy; 16 top-level in fma_medium, 8 balanced in fma_small |
| MTG-Jamendo | 55,701 | 95 genre tags (87 in the official splits), 195 tags overall |
| JamendoMaxCaps | 362,298 | Free-form Jamendo genre tags, partly imputed by an LLM |
| Song Describer | 706 | Through linked MTG-Jamendo metadata |
| SonicMaster dataset | about 25k clips | 10 genre groups defined by the authors |

---

## D. Evaluation practice

### D1. Objective metrics reported in this literature

| Family | Metrics | Papers using them |
|---|---|---|
| Loudness and dynamics | Integrated loudness error (LUFS, ITU-R BS.1770), RMS error, crest factor, loudness range (EBU Tech 3342), dynamic complexity, DRV | DeepAFx-ST (RMS, LUFS); De-limiter (RMS, crest factor, dynamic complexity, LRA, at -14 LUFS); ITO-Master (DRV); Koo 2022 (RMS, side RMS); Mourgela et al. (LUFS, true peak, clipping) |
| Tonal balance | Spectral centroid error, mel-spectral distance, bark-spectrum distance, band energy ratios, LTAS distance | DeepAFx-ST (MSD, SCE); Diff-MST (bark spectrum); SonicMaster (band ratios, 9-band cosine distance); BABE-2 (LTAS distance) |
| Stereo | Side-channel RMS, stereo width, stereo imbalance, mid/side RMS ratio | Koo 2022; Diff-MST; SonicMaster; MEGAMI features |
| Waveform or spectrogram fidelity (paired) | SI-SDR, SDR, SI-SNR, multi-resolution STFT error, SSIM on mel, fw-SNR | RemFX; De-limiter; Apollo; SonicMaster; Koo 2022 |
| Perceptual models (paired) | PEAQ (BS.1387), ViSQOL, PESQ, STOI, Zimtohrli | De-limiter (PEAQ); Apollo (ViSQOL); DeepAFx-ST (PESQ); Koo 2022 (STOI); Karystinaios et al. (Zimtohrli) |
| Distributional | FAD with VGGish, CLAP, DAC, EnCodec embeddings; KAD; KL divergence | Diff-MST (VGGish); ITO-Master (CLAP, DAC, EnCodec); SonicMaster (CLAP, KL); BABE-2 (fadtk, several embeddings); MEGAMI, Diff2Mix, sequential stem blending (KAD) |
| Reference-free quality predictors | Audiobox Aesthetics Production Quality (PQ) | SonicMaster; sequential stem blending; historical restoration (Cho et al.) |
| Style similarity | FXencoder or Fx-Encoder++ cosine similarity, AFx-Rep similarity, audio-feature loss | ITO-Master; ST-ITO; Diff-MST; StemFX |

Metric references: Kilgour et al. 2019 (FAD, DOI 10.21437/Interspeech.2019-2219); Gui et al. 2024 (FAD embedding and sample-size effects, fadtk, DOI 10.1109/ICASSP48485.2024.10446663); Chung et al. 2025 (KAD, arXiv:2502.15602); Tjandra et al. 2025 (Audiobox Aesthetics, arXiv:2502.05139); Chinen et al. 2020 (ViSQOL v3, DOI 10.1109/QoMEX48832.2020.9123150); ITU-R BS.1387-2 (PEAQ, 05/2023); ITU-R BS.1770-5 (loudness and true peak, 11/2023); EBU R 128 v5.0 (11/2023, -23 LUFS programme target).

Known weaknesses that reviewers raise:
- FAD with VGGish runs at 16 kHz mono. De-limiter shows a model with inverted channel polarity scoring badly on FAD for that reason, and conversely stereo or high-band defects can be invisible to it. Yeh et al. (arXiv:2608.05506) note that Audiobox Aesthetics was trained on 16 kHz mono audio.
- Kandpal et al. 2022 measured rank correlation between objective metrics and MOS for music enhancement and found FAD the best aligned of those tested; Vinay and Lerch 2022 (arXiv:2209.00130) and Grotschla et al. 2025 (arXiv:2506.19085) examine metric and human agreement for generative audio.
- Karystinaios et al. 2026 show metric families disagreeing on the same systems and decline to rank without subjective data.
- SonicMaster's own numbers show PQ rising while SSIM falls and FAD barely moves, which illustrates that a generative system can raise a quality predictor while departing from the source.

### D2. Listening test designs in accepted papers

| Paper (venue) | Design | Listeners | Statistics |
|---|---|---|---|
| DeepAFx (ICASSP 2021) | MUSHRA, Web Audio Evaluation Tool, 5 samples per task, 4-s clips, similarity to an engineer's master | 17 (musicians, engineers, critical listeners) | Paired t-tests, Bonferroni |
| DMC (ICASSP 2021) | APE multi-stimulus without reference | 16 audio engineers | Kruskal-Wallis |
| Koo et al. remastering (ICASSP 2022) | MUSHRA, 15 questions, similarity of mastering style to reference | 17 musicians and engineers | Paired t-tests, Bonferroni |
| FxNorm-Automix (ISMIR 2022) | APE, 6 songs x 6 mixes, 25-s, Production Value / Clarity / Excitement, -23 dBFS loudness match, no low anchor | 14 professional mixing engineers (mean 11.6 years) | Pairwise comparisons |
| Music Mixing Style Transfer (ICASSP 2023) | Multi-stimulus similarity, 12 questions | 11 audio engineers (mean 7.3 years) | Paired t-tests, Bonferroni |
| ST-ITO (ISMIR 2024) | Multi-stimulus 0 to 100 similarity, 10 cases, input and oracle included | 23 with audio engineering experience | Descriptive |
| Diff-MST (ISMIR 2024) | None; objective only, stated as a limitation | 0 | n/a |
| ITO-Master (ISMIR 2025) | MUSHRA-type, 8 questions, 30-s stimuli, input as low anchor, no high anchor | 10 with 2 to 5 years of production experience | Pairwise t-tests |
| SonicMaster (ICML 2026) | Study 1: 43 pairs, 7-point Likert on relevance, quality, consistency, preference. Study 2: forced choice among 4 systems, 20 samples | 12 (7 experts, 5 MIR researchers); 20 | Paired t-tests |
| MEGAMI (ICASSP 2026) | Multi-stimulus in isolated booths, 7 songs x 2, 5 stimuli per page | 12 (6 with production experience) | Confidence intervals |
| Diff2Mix (ISMIR 2026) | webMUSHRA online | 20 with professional mixing experience | Pairwise Wilcoxon, Holm |
| StemFX (ISMIR 2026) | MUSHRA-style similarity to target mix | 20 with production experience | Mean scores |
| Text2FX (ICASSP 2025) | Crowd study with hearing screening | 167 after cleaning | Reported per effect |
| Kandpal et al. (ICASSP 2022) | MOS on Mechanical Turk with attention checks | 211; 9,095 answers | MOS, rank correlation |
| Ronan et al. (JAES 2017) | ABX | 20 mastering engineers | Discrimination counts |
| Elliott and Chon (JAES 2022) | Human-vs-machine identification and preference ranking | `[UNVERIFIED]` | `[UNVERIFIED]` |

Tools cited: Web Audio Evaluation Tool (Jillings, De Man, Moffat, Reiss, SMC 2015), APE (De Man and Reiss, 2014), webMUSHRA (Schoeffler et al., DOI 10.5334/jors.187), ITU-R BS.1534-3 (MUSHRA, 10/2015).

### D3. What a strong automatic mastering paper needs in 2026-2027

This subsection is my assessment drawn from the accepted papers above. It is opinion backed by those examples.

1. **Baselines that a reader can rerun.** Matchering and loudness normalization are the floor. ITO-Master set the current comparison set for mastering style transfer (Fx-Normalization, Matchering, E2E Remastering). SonicMaster is the obvious comparison for restoration plus enhancement. A paper that also scores one or two commercial services would go beyond every ML paper since DeepAFx, which compared only against LANDR.
2. **Objective metrics from at least three families.** Loudness and dynamics (integrated LUFS error, LRA, crest factor or PLR, true peak), tonal balance (band or bark spectrum distance), stereo (width or side energy), and one distributional metric with a full-band embedding (CLAP or codec-based FAD, or KAD). ITO-Master reports AF loss, DRV, FX-embedding similarity and FAD under three embeddings. Report measurements after loudness matching, as De-limiter does at -14 LUFS, so that louder outputs do not win by level alone.
3. **Evidence of content preservation.** For any system that resynthesises audio, add paired fidelity numbers (SI-SDR or multi-resolution STFT against the input after gain matching) because reviewers now know that PQ-type scores can rise while SSIM falls (SonicMaster) and that enhancement can degrade inputs (Karystinaios et al.).
4. **A MUSHRA-style or APE-style listening test with experienced listeners.** Accepted mastering and mixing papers used 10 to 23 expert listeners; 16 to 20 is the common range at ICASSP and ISMIR. Include the unprocessed input as an anchor, a human or commercial master as a high reference when one exists, loudness-match all stimuli, and report a test with multiple-comparison correction (Bonferroni in DeepAFx, Koo 2022 and Koo 2023; Holm with Wilcoxon in Diff2Mix; Kruskal-Wallis in DMC). JAES and TASLP reviewers tend to expect BS.1534-style reporting: listener screening, confidence intervals, and per-item results.
5. **Full-band, stereo, full-length evaluation.** 44.1 or 48 kHz stereo is standard since DMC stated it as a requirement. DeepAFx's mastering test at 22 kHz mono would not pass today. Song-level results matter because several systems process 10 to 30-second windows.
6. **Ablations and failure analysis.** ITO-Master ablates black-box vs white-box, encoder training and ITO; SonicMaster ablates model size, solver and conditioning. Per-genre or per-degradation breakdowns are common.
7. **Released code, checkpoints and audio examples.** Every recent accepted system here has a demo page; ISMIR and DAFx reviewers ask about reproducibility explicitly. A released evaluation set would stand out because none exists for mastering.
8. **Venue fit.** ISMIR and ICASSP accept 4 to 6-page papers with one listening test. DAFx and AES conventions value signal-level analysis and interpretable DSP. WASPAA favours a focused method with rigorous objective evaluation (De-limiter, RemFX had no listening test). JAES and TASLP expect broader baselines and generalization tests (DeepAFx-ST in JAES compared neural-proxy, SPSA and autodiff variants against rule-based and end-to-end baselines across speech and music datasets and two sample rates; BABE in TASLP reported both objective and subjective results).

---

## E. Gap analysis

Each item states what I could and could not find. "Not found" means absent from the arXiv, Crossref, OpenAlex, GitHub and web searches run for this report.

1. **No public benchmark of real pre-master and master pairs.** DeepAFx used 138 Mixing Secrets pairs and did not release them. Everything public is synthetic (SonicMaster, musdb-XL) or pairs stems with mastered mixtures (MSRBench). A released, licensed evaluation set, even a small one, would be a first.
2. **No open, reference-free mastering system with interpretable output.** The white-box systems need a reference or a text target (ITO-Master, Matchering, DeepAFx-ST). The reference-free system is a latent generative model (SonicMaster) that resynthesises the whole signal through a VAE. Commercial tools (Ozone Master Assistant, LANDR) choose targets automatically but publish no evaluation. A system that picks its own target and emits DSP parameters is not in the literature I found.
3. **Stem-aware mastering of a finished mix has not been evaluated as a mastering method.** Separation-assisted per-stem processing exists for reference-based mixing style transfer (Koo et al. 2023 with Demucs, StemFX 2026 with pseudo-stems) and for hearing-aid rebalancing (Cadenza). Fx-Encoder++ extracts per-instrument effect embeddings from mixtures. Commercial Ozone has Stem Focus and Master Rebalance. I found no paper that separates a stereo mix, applies restoration or mastering per stem, recombines, and evaluates against whole-mix mastering baselines with artifact measurements.
4. **Restoration and mastering are joined only in SonicMaster, and only generatively.** Its own results show FAD roughly unchanged and SSIM lower than the degraded input, and an independent benchmark finds it weaker on signal-preservation metrics. A combined system that verifiably preserves the source (bounded deviation from the input, measured) is open territory.
5. **No ML mastering paper compares against several commercial services.** DeepAFx compared against LANDR on mono 22 kHz clips; Elliott and Chon compared two services against humans without an ML system. A controlled comparison including Ozone, LANDR and an open baseline at full band would be new.
6. **Delivery-spec compliance is unreported.** Mourgela et al. show most real masters exceed -14 LUFS and many clip. De-limiter evaluates at -14 LUFS. No mastering system paper reports true-peak, LRA and integrated-loudness compliance of its outputs against a stated delivery target.
7. **Whole-song and album-level behaviour.** SonicMaster limits itself to single songs processed in 30-second chunks; Diff-MST states that it is constrained to static mixing configurations; ITO-Master trains on 11.8-second segments and evaluates 30-second excerpts. Time-varying decisions over a full song and consistency across an album are stated as out of scope in those papers.
8. **Full-band stereo quality metrics are weak.** The field's distributional metrics often run at 16 kHz mono (VGGish FAD, Audiobox PQ). A paper that validates a full-band stereo metric against listener ratings for mastering would fill a gap noted by De-limiter and by Yeh et al.
9. **Listening tests are small.** The mastering-specific ML papers used 10 (ITO-Master), 17 (Koo 2022, DeepAFx) and 12 plus 20 (SonicMaster) listeners. A test with 20 or more experienced listeners that includes a human master and a commercial service would exceed current practice.
10. **Local reproducibility of the nearest competitor is limited.** SonicMaster's released inference depends on a gated VAE; Diff-MST has no checkpoint; DeepAFx's mastering model has no advertised checkpoint. A fully open pipeline with ungated weights is itself a differentiator.

Claims to avoid: "first separation-based processing of mixes" (Koo 2023, Mimilakis 2016 WIMP, Cadenza), "first text-controlled mastering" (SonicMaster, ITO-Master CLAP mode), "first neural mastering" (Mimilakis 2016, DeepAFx 2021, Koo 2022), "first de-limiting" (Jeon and Lee 2023).

---

## BibTeX for verified references

Entries carry a `note` when a field (paper number, venue detail, author list) rests on a search snippet or could not be confirmed. Items tagged `[UNVERIFIED]` in the text for their existence are omitted.

```bibtex
% ---------- A1. Mastering ----------
@inproceedings{mimilakis2016dnn,
  author    = {Mimilakis, Stylianos Ioannis and Drossos, Konstantinos and Virtanen, Tuomas and Schuller, Gerald},
  title     = {Deep Neural Networks for Dynamic Range Compression in Mastering Applications},
  booktitle = {Audio Engineering Society Convention 140},
  address   = {Paris, France},
  year      = {2016},
  note      = {Paper 9539 (number from search snippet). AES E-Library 18237}
}
@inproceedings{mimilakis2013tonal,
  author    = {Mimilakis, Stylianos Ioannis and Drossos, Konstantinos and Floros, Andreas and Katerelos, D. T. G.},
  title     = {Automated Tonal Balance Enhancement for Audio Mastering Applications},
  booktitle = {Audio Engineering Society Convention 134},
  address   = {Rome, Italy},
  year      = {2013},
  note      = {AES E-Library 16737}
}
@inproceedings{ma2013yulewalker,
  author    = {Ma, Zheng and Reiss, Joshua D. and Black, Dawn A. A.},
  title     = {Implementation of an Intelligent Equalization Tool Using {Yule-Walker} for Music Mixing and Mastering},
  booktitle = {Audio Engineering Society Convention 134},
  address   = {Rome, Italy},
  year      = {2013},
  note      = {AES E-Library 16792}
}
@inproceedings{martinez2021deepafx,
  author    = {Mart{\'i}nez Ram{\'i}rez, Marco A. and Wang, Oliver and Smaragdis, Paris and Bryan, Nicholas J.},
  title     = {Differentiable Signal Processing With Black-Box Audio Effects},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {66--70},
  year      = {2021},
  doi       = {10.1109/ICASSP39728.2021.9415103},
  eprint    = {2105.04752},
  archivePrefix = {arXiv}
}
@inproceedings{koo2022remastering,
  author    = {Koo, Junghyun and Paik, Seungryeol and Lee, Kyogu},
  title     = {End-to-End Music Remastering System Using Self-Supervised and Adversarial Training},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {4608--4612},
  year      = {2022},
  doi       = {10.1109/ICASSP43922.2022.9746389},
  eprint    = {2202.08520},
  archivePrefix = {arXiv}
}
@inproceedings{koo2025itomaster,
  author    = {Koo, Junghyun and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Fabbro, Giorgio and Mancusi, Michele and Mitsufuji, Yuki},
  title     = {{ITO-Master}: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2025},
  eprint    = {2506.16889},
  archivePrefix = {arXiv}
}
@inproceedings{melechovsky2026sonicmaster,
  author    = {Melechovsky, Jan and Mehrish, Ambuj and Roy, Abhinaba and Herremans, Dorien},
  title     = {{SonicMaster}: Towards Controllable All-in-One Music Restoration and Mastering},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {306},
  address   = {Seoul, South Korea},
  year      = {2026},
  eprint    = {2508.03448},
  archivePrefix = {arXiv}
}
@inproceedings{mourgela2024trends,
  author    = {Mourgela, Angeliki and Quinton, Elio and Bissas, Spyridon and Reiss, Joshua D. and Ronan, David},
  title     = {Exploring Trends in Audio Mixes and Masters: Insights from a Dataset Analysis},
  booktitle = {Audio Engineering Society Convention 157},
  address   = {New York, USA},
  year      = {2024},
  eprint    = {2412.03373},
  archivePrefix = {arXiv}
}
@article{elliott2022comparative,
  author  = {Elliott, Mitchell and Chon, Song Hui},
  title   = {A Comparative Study of Music Mastered by Human Engineers and Automated Services},
  journal = {Journal of the Audio Engineering Society},
  volume  = {70},
  number  = {9},
  pages   = {764--776},
  year    = {2022},
  doi     = {10.17743/jaes.2022.0050}
}
@inproceedings{piotrowska2017instant,
  author    = {Piotrowska, Magdalena and Piotrowski, Szymon and Kostek, Bozena},
  title     = {A Study on Audio Signal Processed by ``Instant Mastering'' Services},
  booktitle = {Audio Engineering Society Convention 142},
  address   = {Berlin, Germany},
  year      = {2017},
  note      = {Paper 9719 (number from search snippet). AES E-Library 18597}
}
@incollection{aker2024aimastering,
  author    = {Aker, Onur},
  title     = {{AI}-Assisted Music Mastering},
  booktitle = {Understanding Generative AI in a Cultural Context},
  publisher = {IGI Global},
  pages     = {17--50},
  year      = {2024},
  doi       = {10.4018/979-8-3693-7235-7.ch002},
  note      = {Book title as reported by a helper agent; Crossref lists the series Advances in Human and Social Aspects of Technology}
}
@article{sterne2019landr,
  author  = {Sterne, Jonathan and Razlogova, Elena},
  title   = {Machine Learning in Context, or Learning from {LANDR}: Artificial Intelligence and the Platformization of Music Mastering},
  journal = {Social Media + Society},
  volume  = {5},
  number  = {2},
  year    = {2019},
  doi     = {10.1177/2056305119847525}
}
@article{sterne2021tuning,
  author  = {Sterne, Jonathan and Razlogova, Elena},
  title   = {Tuning Sound for Infrastructures: Artificial Intelligence, Automation, and the Cultural Politics of Audio Mastering},
  journal = {Cultural Studies},
  volume  = {35},
  number  = {4-5},
  pages   = {750--770},
  year    = {2021},
  doi     = {10.1080/09502386.2021.1895247}
}
@article{birtchnell2018automating,
  author  = {Birtchnell, Thomas and Elliott, Anthony},
  title   = {Automating the Black Art: Creative Places for Artificial Intelligence in Audio Mastering},
  journal = {Geoforum},
  volume  = {96},
  pages   = {77--86},
  year    = {2018},
  doi     = {10.1016/j.geoforum.2018.08.005}
}
@article{birtchnell2018listening,
  author  = {Birtchnell, Thomas},
  title   = {Listening Without Ears: Artificial Intelligence in Audio Mastering},
  journal = {Big Data \& Society},
  volume  = {5},
  number  = {2},
  year    = {2018},
  doi     = {10.1177/2053951718808553}
}
@article{collins2021mastering,
  author  = {Collins, Steve and Renzo, Adrian and Keith, Sarah and Mesker, Alex},
  title   = {Mastering 2.0: The Real or Perceived Threat of {DIY} Mastering and Automated Mastering Systems},
  journal = {Popular Music and Society},
  volume  = {44},
  number  = {3},
  pages   = {258--273},
  year    = {2021},
  doi     = {10.1080/03007766.2019.1699339},
  note    = {Published online 2019; issue dated 2021}
}

% ---------- A2. Style transfer and differentiable effects ----------
@article{steinmetz2022deepafxst,
  author  = {Steinmetz, Christian J. and Bryan, Nicholas J. and Reiss, Joshua D.},
  title   = {Style Transfer of Audio Effects with Differentiable Signal Processing},
  journal = {Journal of the Audio Engineering Society},
  volume  = {70},
  number  = {9},
  pages   = {708--721},
  year    = {2022},
  doi     = {10.17743/jaes.2022.0025},
  eprint  = {2207.08759},
  archivePrefix = {arXiv}
}
@inproceedings{steinmetz2024stito,
  author    = {Steinmetz, Christian J. and Singh, Shubhr and Comunit{\`a}, Marco and Ibnyahya, Ilias and Yuan, Shanxin and Benetos, Emmanouil and Reiss, Joshua D.},
  title     = {{ST-ITO}: Controlling Audio Effects for Style Transfer with Inference-Time Optimization},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2024},
  eprint    = {2410.21233},
  archivePrefix = {arXiv}
}
@inproceedings{mimilakis2020oneshot,
  author    = {Mimilakis, Stylianos I. and Bryan, Nicholas J. and Smaragdis, Paris},
  title     = {One-Shot Parametric Audio Production Style Transfer with Application to Frequency Equalization},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {256--260},
  year      = {2020},
  doi       = {10.1109/ICASSP40776.2020.9054108}
}
@article{colonel2021reverse,
  author  = {Colonel, Joseph T. and Reiss, Joshua},
  title   = {Reverse Engineering of a Recording Mix with Differentiable Digital Signal Processing},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {150},
  number  = {1},
  pages   = {608--619},
  year    = {2021},
  doi     = {10.1121/10.0005622}
}
@misc{comunita2025nablafx,
  author = {Comunit{\`a}, Marco and Steinmetz, Christian J. and Reiss, Joshua D.},
  title  = {{NablAFx}: A Framework for Differentiable Black-box and Gray-box Modeling of Audio Effects},
  year   = {2025},
  eprint = {2502.11668},
  archivePrefix = {arXiv}
}

% ---------- A3. Automatic mixing ----------
@inproceedings{steinmetz2021dmc,
  author    = {Steinmetz, Christian J. and Pons, Jordi and Pascual, Santiago and Serr{\`a}, Joan},
  title     = {Automatic Multitrack Mixing with a Differentiable Mixing Console of Neural Audio Effects},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {71--75},
  year      = {2021},
  doi       = {10.1109/ICASSP39728.2021.9414364},
  eprint    = {2010.10291},
  archivePrefix = {arXiv}
}
@inproceedings{martinez2022fxnorm,
  author    = {Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Fabbro, Giorgio and Uhlich, Stefan and Nagashima, Chihiro and Mitsufuji, Yuki},
  title     = {Automatic Music Mixing with Deep Learning and Out-of-Domain Data},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2022},
  eprint    = {2208.11428},
  archivePrefix = {arXiv}
}
@inproceedings{koo2023mst,
  author    = {Koo, Junghyun and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Uhlich, Stefan and Lee, Kyogu and Mitsufuji, Yuki},
  title     = {Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10096458},
  eprint    = {2211.02247},
  archivePrefix = {arXiv}
}
@inproceedings{vanka2024diffmst,
  author    = {Vanka, Soumya Sai and Steinmetz, Christian and Rolland, Jean-Baptiste and Reiss, Joshua and Fazekas, Gy{\"o}rgy},
  title     = {{Diff-MST}: Differentiable Mixing Style Transfer},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2024},
  eprint    = {2407.08889},
  archivePrefix = {arXiv}
}
@misc{vanka2024diffmstc,
  author = {Vanka, Soumya Sai and Hannink, Lennart and Rolland, Jean-Baptiste and Fazekas, George},
  title  = {{Diff-MSTC}: A Mixing Style Transfer Prototype for {Cubase}},
  year   = {2024},
  eprint = {2411.06576},
  archivePrefix = {arXiv}
}
@inproceedings{moliner2026megami,
  author    = {Moliner, Eloi and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Koo, Junghyun and Liao, Wei-Hsiang and Cheuk, Kin Wai and Serr{\`a}, Joan and V{\"a}lim{\"a}ki, Vesa and Mitsufuji, Yuki},
  title     = {Automatic Music Mixing Using a Generative Model of Effect Embeddings},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {14582--14586},
  year      = {2026},
  doi       = {10.1109/ICASSP55912.2026.11462677},
  eprint    = {2511.08040},
  archivePrefix = {arXiv}
}
@inproceedings{zong2026diff2mix,
  author    = {Zong, Yisu and Shi, Jinjie and Reiss, Joshua},
  title     = {{Diff2Mix}: Controllable Music Mixing via Diffusion Models and Differentiable Audio Effects},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  eprint    = {2608.05442},
  archivePrefix = {arXiv}
}
@misc{yeh2026sequential,
  author = {Yeh, Yen-Tung and Chan, Chung-Jui and Hung, Yun-Ning and Yang, Yi-Hsuan},
  title  = {Rethinking Automatic Music Mixing as Sequential Stem Blending},
  year   = {2026},
  eprint = {2608.05506},
  archivePrefix = {arXiv}
}
@inproceedings{liu2026relfx,
  author    = {Liu, Xinlu and Lin, Huibin and Wei, Weixing and Yan, Zhenhai},
  title     = {Beyond Dry References: Learning Relative Audio Effects Representations via Contrastive Distance Learning},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  eprint    = {2608.10573},
  archivePrefix = {arXiv}
}
@inproceedings{yu2025diffvox,
  author    = {Yu, Chin-Yun and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Koo, Junghyun and Hayes, Ben and Liao, Wei-Hsiang and Fazekas, Gy{\"o}rgy and Mitsufuji, Yuki},
  title     = {{DiffVox}: A Differentiable Model for Capturing and Analysing Vocal Effects Distributions},
  booktitle = {Proc. International Conference on Digital Audio Effects (DAFx)},
  year      = {2025},
  eprint    = {2504.14735},
  archivePrefix = {arXiv}
}
@article{ma2015intelligent,
  author  = {Ma, Zheng and De Man, Brecht and Pestana, Pedro D. L. and Black, Dawn A. A. and Reiss, Joshua D.},
  title   = {Intelligent Multitrack Dynamic Range Compression},
  journal = {Journal of the Audio Engineering Society},
  pages   = {412--426},
  year    = {2015},
  doi     = {10.17743/jaes.2015.0053},
  note    = {Crossref returns no volume or issue}
}
@article{hafezi2015autonomous,
  author  = {Hafezi, Sina and Reiss, Joshua},
  title   = {Autonomous Multitrack Equalization Based on Masking Reduction},
  journal = {Journal of the Audio Engineering Society},
  volume  = {63},
  number  = {5},
  pages   = {312--323},
  year    = {2015},
  doi     = {10.17743/jaes.2015.0021}
}
@article{martinez2021drummixing,
  author  = {Mart{\'i}nez Ram{\'i}rez, Marco A. and Stoller, Daniel and Moffat, David},
  title   = {A Deep Learning Approach to Intelligent Drum Mixing With the {Wave-U-Net}},
  journal = {Journal of the Audio Engineering Society},
  volume  = {69},
  number  = {3},
  pages   = {142--151},
  year    = {2021},
  doi     = {10.17743/jaes.2020.0031}
}
@inproceedings{mockenhaupt2024autoeq,
  author    = {Mockenhaupt, Florian and Rieber, Joscha Simon and Nercessian, Shahan},
  title     = {Automatic Equalization for Individual Instrument Tracks Using Convolutional Neural Networks},
  booktitle = {Proc. International Conference on Digital Audio Effects (DAFx)},
  year      = {2024},
  eprint    = {2407.16691},
  archivePrefix = {arXiv}
}
@book{deman2019imp,
  author    = {De Man, Brecht and Stables, Ryan and Reiss, Joshua D.},
  title     = {Intelligent Music Production},
  publisher = {Routledge},
  year      = {2019},
  doi       = {10.4324/9781315166100}
}
@inproceedings{deman2017tenyears,
  author    = {De Man, Brecht and Reiss, Joshua D. and Stables, Ryan},
  title     = {Ten Years of Automatic Mixing},
  booktitle = {Proc. 3rd Workshop on Intelligent Music Production (WIMP)},
  year      = {2017}
}
@article{moffat2019approaches,
  author  = {Moffat, David and Sandler, Mark B.},
  title   = {Approaches in Intelligent Music Production},
  journal = {Arts},
  volume  = {8},
  number  = {4},
  pages   = {125},
  year    = {2019},
  doi     = {10.3390/arts8040125}
}
@phdthesis{pestana2013thesis,
  author = {Pestana, Pedro Duarte Leal Gomes},
  title  = {Automatic Mixing Systems Using Adaptive Digital Audio Effects},
  school = {Universidade Cat{\'o}lica Portuguesa},
  year   = {2013},
  url    = {http://hdl.handle.net/10400.14/10887}
}
@inproceedings{vanka2023adoption,
  author    = {Vanka, Soumya Sai and Safi, Maryam and Rolland, Jean-Baptiste and Fazekas, George},
  title     = {Adoption of {AI} Technology in the Music Mixing Workflow: An Investigation},
  booktitle = {Audio Engineering Society Convention 154},
  year      = {2023},
  note      = {Paper 10653},
  eprint    = {2304.03407},
  archivePrefix = {arXiv}
}
@misc{vanka2023role,
  author = {Vanka, Soumya Sai and Safi, Maryam and Rolland, Jean-Baptiste and Fazekas, Gy{\"o}rgy},
  title  = {The Role of Communication and Reference Songs in the Mixing Process: Insights from Professional Mix Engineers},
  year   = {2023},
  eprint = {2309.03404},
  archivePrefix = {arXiv}
}

% ---------- A4. Blind estimation and removal ----------
@inproceedings{jeon2023delimiter,
  author    = {Jeon, Chang-Bin and Lee, Kyogu},
  title     = {Music De-limiter Networks via Sample-wise Gain Inversion},
  booktitle = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year      = {2023},
  doi       = {10.1109/WASPAA58266.2023.10248055},
  eprint    = {2308.01187},
  archivePrefix = {arXiv}
}
@inproceedings{jeon2022musdbxl,
  author    = {Jeon, Chang-Bin and Lee, Kyogu},
  title     = {Towards Robust Music Source Separation on Loud Commercial Music},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2022},
  eprint    = {2208.14355},
  archivePrefix = {arXiv}
}
@inproceedings{rice2023remfx,
  author    = {Rice, Matthew and Steinmetz, Christian J. and Fazekas, George and Reiss, Joshua D.},
  title     = {General Purpose Audio Effect Removal},
  booktitle = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year      = {2023},
  doi       = {10.1109/WASPAA58266.2023.10248157},
  eprint    = {2308.16177},
  archivePrefix = {arXiv}
}
@inproceedings{imort2022distortion,
  author    = {Imort, Johannes and Fabbro, Giorgio and Mart{\'i}nez Ram{\'i}rez, Marco A. and Uhlich, Stefan and Koyama, Yuichiro and Mitsufuji, Yuki},
  title     = {Distortion Audio Effects: Learning How to Recover the Clean Signal},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2022},
  eprint    = {2202.01664},
  archivePrefix = {arXiv},
  note      = {ISMIR 2022 venue as cited by later papers; arXiv record has no venue comment}
}
@inproceedings{peladeau2024blind,
  author    = {Peladeau, C{\^o}me and Peeters, Geoffroy},
  title     = {Blind Estimation of Audio Effects Using an Auto-Encoder Approach and Differentiable Digital Signal Processing},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {856--860},
  year      = {2024},
  doi       = {10.1109/ICASSP48485.2024.10448301},
  eprint    = {2310.11781},
  archivePrefix = {arXiv}
}
@inproceedings{lee2023blindgraph,
  author    = {Lee, Sungho and Park, Jaehyun and Paik, Seungryeol and Lee, Kyogu},
  title     = {Blind Estimation of Audio Processing Graph},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10096581},
  eprint    = {2303.08610},
  archivePrefix = {arXiv}
}
@inproceedings{lee2024pruning,
  author    = {Lee, Sungho and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Uhlich, Stefan and Fabbro, Giorgio and Lee, Kyogu and Mitsufuji, Yuki},
  title     = {Searching For Music Mixing Graphs: A Pruning Approach},
  booktitle = {Proc. International Conference on Digital Audio Effects (DAFx)},
  year      = {2024},
  eprint    = {2406.01049},
  archivePrefix = {arXiv}
}
@article{lee2025reverse,
  author  = {Lee, Sungho and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Uhlich, Stefan and Fabbro, Giorgio and Lee, Kyogu and Mitsufuji, Yuki},
  title   = {Reverse Engineering of Music Mixing Graphs With Differentiable Processors and Iterative Pruning},
  journal = {Journal of the Audio Engineering Society},
  volume  = {73},
  number  = {6},
  pages   = {344--365},
  year    = {2025},
  doi     = {10.17743/jaes.2022.0212},
  eprint  = {2509.15948},
  archivePrefix = {arXiv}
}
@misc{sun2024drcinversion,
  author = {Sun, Haoran and Fourer, Dominique and Maaref, Hichem},
  title  = {Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics},
  year   = {2024},
  eprint = {2411.04337},
  archivePrefix = {arXiv}
}
@misc{sun2026blackbox,
  author = {Sun, Haoran and Fourer, Dominique and Maaref, Hichem},
  title  = {Black-Box Optimization for Identifying and Inverting Audio Dynamic Range Control Effects},
  year   = {2026},
  eprint = {2607.19645},
  archivePrefix = {arXiv}
}
@article{hinrichs2022cnn,
  author  = {Hinrichs, Reemt and Gerkens, Kevin and Lange, Alexander and Ostermann, J{\"o}rn},
  title   = {Convolutional Neural Networks for the Classification of Guitar Effects and Extraction of the Parameter Settings of Single and Multi-Guitar Effects from Instrument Mixes},
  journal = {EURASIP Journal on Audio, Speech, and Music Processing},
  volume  = {2022},
  number  = {1},
  year    = {2022},
  doi     = {10.1186/s13636-022-00257-4}
}
@article{hinrichs2024blind,
  author  = {Hinrichs, Reemt and Gerkens, Kevin and Lange, Alexander and Ostermann, J{\"o}rn},
  title   = {Blind Extraction of Guitar Effects Through Blind System Inversion and Neural Guitar Effect Modeling},
  journal = {EURASIP Journal on Audio, Speech, and Music Processing},
  volume  = {2024},
  number  = {1},
  year    = {2024},
  doi     = {10.1186/s13636-024-00330-0}
}
@misc{yang2025wildfx,
  author = {Yang, Qihui and Berg-Kirkpatrick, Taylor and McAuley, Julian and Novack, Zachary},
  title  = {{WildFX}: A {DAW}-Powered Pipeline for In-the-Wild Audio {FX} Graph Modeling},
  year   = {2025},
  eprint = {2507.10534},
  archivePrefix = {arXiv}
}

% ---------- A5. Restoration ----------
@inproceedings{li2025apollo,
  author    = {Li, Kai and Luo, Yi},
  title     = {Apollo: Band-sequence Modeling for High-Quality Audio Restoration},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10890825},
  eprint    = {2409.08514},
  archivePrefix = {arXiv}
}
@article{moliner2024babe,
  author  = {Moliner, Eloi and Elvander, Filip and V{\"a}lim{\"a}ki, Vesa},
  title   = {Blind Audio Bandwidth Extension: A Diffusion-Based Zero-Shot Approach},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume  = {32},
  pages   = {5092--5105},
  year    = {2024},
  doi     = {10.1109/TASLP.2024.3507566},
  eprint  = {2306.01433},
  archivePrefix = {arXiv}
}
@inproceedings{moliner2024babe2,
  author    = {Moliner, Eloi and Turunen, Maija and Elvander, Filip and V{\"a}lim{\"a}ki, Vesa},
  title     = {A Diffusion-Based Generative Equalizer for Music Restoration},
  booktitle = {Proc. 27th International Conference on Digital Audio Effects (DAFx24)},
  address   = {Guildford, UK},
  year      = {2024},
  eprint    = {2403.18636},
  archivePrefix = {arXiv}
}
@article{moliner2023behmgan,
  author  = {Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa},
  title   = {{BEHM-GAN}: Bandwidth Extension of Historical Music Using Generative Adversarial Networks},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume  = {31},
  pages   = {943--956},
  year    = {2023},
  doi     = {10.1109/TASLP.2022.3190726}
}
@inproceedings{kandpal2022musicenhancement,
  author    = {Kandpal, Nikhil and Nieto, Oriol and Jin, Zeyu},
  title     = {Music Enhancement via Image Translation and Vocoding},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {3124--3128},
  year      = {2022},
  doi       = {10.1109/ICASSP43922.2022.9747454},
  eprint    = {2204.13289},
  archivePrefix = {arXiv}
}
@misc{zang2025msr,
  author = {Zang, Yongyi and Dai, Zheqi and Plumbley, Mark D. and Kong, Qiuqiang},
  title  = {Music Source Restoration},
  year   = {2025},
  eprint = {2505.21827},
  archivePrefix = {arXiv}
}
@misc{zang2025msrbench,
  author = {Zang, Yongyi and Hai, Jiarui and Ge, Wanying and Kong, Qiuqiang and Dai, Zheqi and Wang, Helin and Mitsufuji, Yuki and Plumbley, Mark D.},
  title  = {{MSRBench}: A Benchmarking Dataset for Music Source Restoration},
  year   = {2025},
  eprint = {2510.10995},
  archivePrefix = {arXiv}
}
@misc{zang2026msrchallenge,
  author = {Zang, Yongyi and Hai, Jiarui and Ge, Wanying and Kong, Qiuqiang and Dai, Zheqi and Wang, Helin and Mitsufuji, Yuki and Plumbley, Mark D.},
  title  = {Summary of The Inaugural Music Source Restoration Challenge},
  year   = {2026},
  eprint = {2601.04343},
  archivePrefix = {arXiv}
}
@inproceedings{svento2026loudar,
  author    = {{\v S}vento, Michal and Moliner, Eloi and Kallinen, Valtteri and Juvela, Lauri and V{\"a}lim{\"a}ki, Vesa and Rajmic, Pavel},
  title     = {Music Restoration via Latent Operator Optimization and Diffusion Model Priors},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  eprint    = {2608.01972},
  archivePrefix = {arXiv}
}
@misc{cho2026historical,
  author = {Cho, Steven and Koo, Junghyun and Lafargue, Raphael and Dhyani, Tushar and Moliner, Eloi and Mitsufuji, Yuki},
  title  = {End-to-End Historical Music Restoration in Latent Space},
  year   = {2026},
  eprint = {2610.00607},
  archivePrefix = {arXiv}
}
@misc{karystinaios2026lowlatency,
  author = {Karystinaios, Emmanouil and Greif, Jonathan and Nadrchal, David and Primus, Paul and Widmer, Gerhard},
  title  = {Low-Latency Neural Models for Real-Time Music Enhancement},
  year   = {2026},
  eprint = {2607.12872},
  archivePrefix = {arXiv}
}
@inproceedings{liu2026dsme,
  author    = {Liu, Fei and Ai, Yang and Ling, Zhen-Hua},
  title     = {Neural Music Enhancement with Dual Time-Frequency Spectral Representations for Prediction and Discrimination},
  booktitle = {Proc. ISCSLP},
  year      = {2026},
  eprint    = {2609.03357},
  archivePrefix = {arXiv}
}
@article{lemercier2024diffusion,
  author  = {Lemercier, Jean-Marie and Richter, Julius and Welker, Simon and Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa and Gerkmann, Timo},
  title   = {Diffusion Models for Audio Restoration: A Review},
  journal = {IEEE Signal Processing Magazine},
  volume  = {41},
  number  = {6},
  pages   = {72--84},
  year    = {2024},
  doi     = {10.1109/MSP.2024.3445871}
}

% ---------- A6. Separation-assisted remixing ----------
@inproceedings{yang2022remix,
  author    = {Yang, Haici and Firodiya, Shivani and Bryan, Nicholas J. and Kim, Minje},
  title     = {Don't Separate, Learn to Remix: End-to-End Neural Remixing with Joint Optimization},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {116--120},
  year      = {2022},
  doi       = {10.1109/ICASSP43922.2022.9746077},
  eprint    = {2107.13634},
  archivePrefix = {arXiv}
}
@inproceedings{wierstorf2017perceptual,
  author    = {Wierstorf, Hagen and Ward, Dominic and Mason, Russell and Grais, Emad M. and Hummersone, Chris and Plumbley, Mark D.},
  title     = {Perceptual Evaluation of Source Separation for Remixing Music},
  booktitle = {Audio Engineering Society Convention 143},
  address   = {New York, USA},
  year      = {2017},
  note      = {AES E-Library 19277. Author list beyond Wierstorf and Ward not re-verified. Stimuli: doi:10.5281/zenodo.835182}
}
@inproceedings{roma2016remixing,
  author    = {Roma, Gerard and Grais, Emad M. and Simpson, Andrew J. R. and Plumbley, Mark D.},
  title     = {Music Remixing and Upmixing Using Source Separation},
  booktitle = {Proc. 2nd Workshop on Intelligent Music Production (WIMP)},
  year      = {2016},
  note      = {Venue per University of Surrey repository record}
}
@misc{simpson2015deepremix,
  author = {Simpson, Andrew J. R. and Roma, Gerard and Plumbley, Mark D.},
  title  = {Deep Remix: Remixing Musical Mixtures Using a Convolutional Deep Neural Network},
  year   = {2015},
  eprint = {1505.00289},
  archivePrefix = {arXiv}
}
@inproceedings{mimilakis2016jazz,
  author    = {Mimilakis, Stylianos Ioannis and Cano, Estefan{\'i}a and Abe{\ss}er, Jakob and Schuller, Gerald},
  title     = {New Sonorities for Jazz Recordings: Separation and Mixing Using Deep Neural Networks},
  booktitle = {Proc. 2nd Workshop on Intelligent Music Production (WIMP)},
  year      = {2016},
  note      = {Record from Fraunhofer Publica via search snippet}
}
@article{pons2016remixing,
  author  = {Pons, Jordi and Janer, Jordi and Rode, Thilo and Nogueira, Waldo},
  title   = {Remixing Music Using Source Separation Algorithms to Improve the Musical Experience of Cochlear Implant Users},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {140},
  number  = {6},
  pages   = {4338--4349},
  year    = {2016},
  doi     = {10.1121/1.4971424}
}
@misc{roadabike2024cadenza,
  author = {Roa Dabike, Gerardo and Akeroyd, Michael A. and Bannister, Scott and Barker, Jon P. and Cox, Trevor J. and Fazenda, Bruno and Firth, Jennifer and Graetzer, Simone and Greasley, Alinka and Vos, Rebecca R. and Whitmer, William M.},
  title  = {The First {Cadenza} Challenges: Using Machine Learning Competitions to Improve Music for Listeners with a Hearing Loss},
  year   = {2024},
  eprint = {2409.05095},
  archivePrefix = {arXiv}
}
@inproceedings{roadabike2024icassp,
  author    = {Roa Dabike, Gerardo and Akeroyd, Michael A. and Bannister, Scott and Barker, Jon and Cox, Trevor J. and Fazenda, Bruno and Firth, Jennifer and Graetzer, Simone and Greasley, Alinka and Vos, Rebecca R. and Whitmer, William M.},
  title     = {The {ICASSP SP Cadenza} Challenge: Music Demixing/Remixing for Hearing Aids},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  year      = {2024},
  doi       = {10.1109/ICASSPW62465.2024.10626340}
}
@article{bannister2026cadenza,
  author  = {Bannister, Scott and Firth, Jennifer and Roa-Dabike, Gerardo and Vos, Rebecca and Whitmer, William and Greasley, Alinka E. and Graetzer, Simone and Fazenda, Bruno and Cox, Trevor and Barker, Jon and Akeroyd, Michael A.},
  title   = {The First {Cadenza} Challenge: Perceptual Evaluation of Machine Learning Systems to Improve Audio Quality of Popular Music for Those with Hearing Loss},
  journal = {Trends in Hearing},
  volume  = {30},
  year    = {2026},
  doi     = {10.1177/23312165251408761}
}
@inproceedings{yeh2025fxencoderpp,
  author    = {Yeh, Yen-Tung and Koo, Junghyun and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Yang, Yi-Hsuan and Mitsufuji, Yuki},
  title     = {{Fx-Encoder++}: Extracting Instrument-Wise Audio Effects Representations from Mixtures},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2025},
  eprint    = {2507.02273},
  archivePrefix = {arXiv}
}
@inproceedings{cheng2026stemfx,
  author    = {Cheng, Yuan-Chiao and Wu, Jui-Te and Chen, Brian and Yeh, Yen-Tung and Chen, Yu-Hua and Yang, Yi-Hsuan},
  title     = {{StemFX}: Learning Mixing Style Representations via Autoregressive {FX} Chain Prediction on Source-Separated Stems},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  eprint    = {2607.15634},
  archivePrefix = {arXiv}
}
@inproceedings{guo2026sphere,
  author    = {Guo, Zixun and Murdock, Calvin and Parekh, Sanjeel and Brimijoin, W. Owen and Dixon, Simon and Reiss, Joshua and Ananthabhotla, Ishwarya},
  title     = {{SPHERE}: Automatic Music Upmixing via Audio Language Model Post-Training with Spatial Heuristic Rewards},
  booktitle = {Proc. EMNLP},
  year      = {2026},
  eprint    = {2608.30559},
  archivePrefix = {arXiv}
}
@article{cano2019mss,
  author  = {Cano, Estefan{\'i}a and FitzGerald, Derry and Liutkus, Antoine and Plumbley, Mark D. and St{\"o}ter, Fabian-Robert},
  title   = {Musical Source Separation: An Introduction},
  journal = {IEEE Signal Processing Magazine},
  volume  = {36},
  number  = {1},
  pages   = {31--40},
  year    = {2019},
  doi     = {10.1109/MSP.2018.2874719}
}

% ---------- A7. Loudness, spectra, perception ----------
@article{deruty2014dynamic,
  author  = {Deruty, Emmanuel and Tardieu, Damien},
  title   = {About Dynamic Processing in Mainstream Music},
  journal = {Journal of the Audio Engineering Society},
  volume  = {62},
  number  = {1/2},
  pages   = {42--55},
  year    = {2014},
  doi     = {10.17743/jaes.2014.0001}
}
@inproceedings{deruty2015mir,
  author    = {Deruty, Emmanuel and Pachet, Fran{\c c}ois},
  title     = {The {MIR} Perspective on the Evolution of Dynamics in Mainstream Music},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2015}
}
@inproceedings{vickers2010loudness,
  author    = {Vickers, Earl},
  title     = {The Loudness War: Background, Speculation, and Recommendations},
  booktitle = {Audio Engineering Society Convention 129},
  year      = {2010}
}
@article{serra2012evolution,
  author  = {Serr{\`a}, Joan and Corral, {\'A}lvaro and Bogu{\~n}{\'a}, Mari{\'a}n and Haro, Mart{\'i}n and Arcos, Josep Ll.},
  title   = {Measuring the Evolution of Contemporary Western Popular Music},
  journal = {Scientific Reports},
  volume  = {2},
  pages   = {521},
  year    = {2012},
  doi     = {10.1038/srep00521}
}
@article{hove2019bass,
  author  = {Hove, Michael J. and Vuust, Peter and Stupacher, Jan},
  title   = {Increased Levels of Bass in Popular Music Recordings 1955--2016 and Their Relation to Loudness},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {145},
  number  = {4},
  pages   = {2247--2253},
  year    = {2019},
  doi     = {10.1121/1.5097587}
}
@inproceedings{pestana2013spectral,
  author    = {Pestana, Pedro Duarte and Ma, Zheng and Reiss, Joshua D. and Barbosa, {\'A}lvaro and Black, Dawn A. A.},
  title     = {Spectral Characteristics of Popular Commercial Recordings 1950-2010},
  booktitle = {Audio Engineering Society Convention 135},
  address   = {New York, USA},
  year      = {2013},
  note      = {Paper 8960 (number from search snippet). AES E-Library 17010}
}
@article{hjortkjaer2014perceptual,
  author  = {Hjortkj{\ae}r, Jens and Walther-Hansen, Mads},
  title   = {Perceptual Effects of Dynamic Range Compression in Popular Music Recordings},
  journal = {Journal of the Audio Engineering Society},
  volume  = {62},
  number  = {1/2},
  pages   = {37--41},
  year    = {2014},
  doi     = {10.17743/jaes.2014.0003}
}
@article{croghan2012quality,
  author  = {Croghan, Naomi B. H. and Arehart, Kathryn H. and Kates, James M.},
  title   = {Quality and Loudness Judgments for Music Subjected to Compression Limiting},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {132},
  number  = {2},
  pages   = {1177--1188},
  year    = {2012},
  doi     = {10.1121/1.4730881}
}
@article{ronan2017hypercompression,
  author  = {Ronan, Malachy and Ward, Nicholas and Sazdov, Robert and Lee, Hyunkook},
  title   = {The Perception of Hyper-Compression by Mastering Engineers},
  journal = {Journal of the Audio Engineering Society},
  volume  = {65},
  number  = {7/8},
  pages   = {613--621},
  year    = {2017},
  doi     = {10.17743/jaes.2017.0023}
}
@article{wilson2016perception,
  author  = {Wilson, Alex and Fazenda, Bruno},
  title   = {Perception of Audio Quality in Productions of Popular Music},
  journal = {Journal of the Audio Engineering Society},
  volume  = {64},
  number  = {1/2},
  pages   = {23--34},
  year    = {2016},
  doi     = {10.17743/jaes.2015.0090}
}
@article{wilson2016variation,
  author  = {Wilson, Alex and Fazenda, Bruno},
  title   = {Variation in Multitrack Mixes: Analysis of Low-level Audio Signal Features},
  journal = {Journal of the Audio Engineering Society},
  volume  = {64},
  number  = {7/8},
  pages   = {466--473},
  year    = {2016},
  doi     = {10.17743/jaes.2016.0029}
}
@inproceedings{wilson2015mixes,
  author    = {Wilson, Alex and Fazenda, Bruno},
  title     = {101 Mixes: A Statistical Analysis of Mix-Variation in a Dataset of Multi-Track Music Mixes},
  booktitle = {Audio Engineering Society Convention 139},
  year      = {2015}
}
@inproceedings{deman2017mixeval,
  author    = {De Man, Brecht and Reiss, Joshua D.},
  title     = {The Mix Evaluation Dataset},
  booktitle = {Proc. 20th International Conference on Digital Audio Effects (DAFx-17)},
  address   = {Edinburgh, UK},
  year      = {2017}
}

% ---------- A8. Text control ----------
@article{venkatesh2022word,
  author  = {Venkatesh, Satvik and Moffat, David and Miranda, Eduardo Reck},
  title   = {Word Embeddings for Automatic Equalization in Audio Mixing},
  journal = {Journal of the Audio Engineering Society},
  volume  = {70},
  number  = {9},
  pages   = {753--763},
  year    = {2022},
  doi     = {10.17743/jaes.2022.0047},
  eprint  = {2202.08898},
  archivePrefix = {arXiv}
}
@inproceedings{zheng2016socialfx,
  author    = {Zheng, Taylor and Seetharaman, Prem and Pardo, Bryan},
  title     = {{SocialFX}: Studying a Crowdsourced Folksonomy of Audio Effects Terms},
  booktitle = {Proc. 24th ACM International Conference on Multimedia},
  pages     = {182--186},
  year      = {2016},
  doi       = {10.1145/2964284.2967207}
}
@inproceedings{chu2025text2fx,
  author    = {Chu, Annie and O'Reilly, Patrick and Barnett, Julia and Pardo, Bryan},
  title     = {{Text2FX}: Harnessing {CLAP} Embeddings for Text-Guided Audio Effects},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10890334},
  eprint    = {2409.18847},
  archivePrefix = {arXiv}
}
@inproceedings{doh2025llm2fx,
  author    = {Doh, Seungheon and Koo, Junghyun and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Liao, Wei-Hsiang and Nam, Juhan and Mitsufuji, Yuki},
  title     = {Can Large Language Models Predict Audio Effects Parameters from Natural Language?},
  booktitle = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year      = {2025},
  doi       = {10.1109/WASPAA66052.2025.11230953},
  eprint    = {2505.20770},
  archivePrefix = {arXiv}
}
@inproceedings{doh2026llm2fxtools,
  author    = {Doh, Seungheon and Koo, Junghyun and Mart{\'i}nez-Ram{\'i}rez, Marco A. and Choi, Woosung and Liao, Wei-Hsiang and Wu, Qiyu and Nam, Juhan and Mitsufuji, Yuki},
  title     = {{LLM2Fx-Tools}: Tool Calling For Music Post-Production},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2512.01559},
  archivePrefix = {arXiv}
}
@inproceedings{clemens2025mixassist,
  author    = {Clemens, Michael and Marasovi{\'c}, Ana},
  title     = {{MixAssist}: An Audio-Language Dataset for Co-Creative {AI} Assistance in Music Mixing},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025},
  eprint    = {2507.06329},
  archivePrefix = {arXiv}
}
@misc{schaffer2026rime,
  author = {Schaffer, Noah and Singh, Nikhil},
  title  = {{RIME}: Enabling Large-Scale Agentic Music Post-Production},
  year   = {2026},
  eprint = {2607.19605},
  archivePrefix = {arXiv}
}
@inproceedings{yu2026instructfx2fx,
  author    = {Yu, Song-Ze and Liessens Dujardin, Milan and Cai, Yuxuan and Zhang, Wantong and Cruz, Brian and Wagner, Jeremy and Cella, Carmine-Emanuele},
  title     = {{InstructFX2FX}: A Multi-Turn Text-to-Effect System for Sequential Audio Effect Refinement},
  booktitle = {Proc. 29th International Conference on Digital Audio Effects (DAFx26)},
  address   = {Cambridge, MA, USA},
  pages     = {526--529},
  year      = {2026},
  eprint    = {2606.22005},
  archivePrefix = {arXiv}
}

% ---------- D. Metrics and listening-test tools ----------
@inproceedings{kilgour2019fad,
  author    = {Kilgour, Kevin and Zuluaga, Mauricio and Roblek, Dominik and Sharifi, Matthew},
  title     = {Fr{\'e}chet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms},
  booktitle = {Proc. Interspeech},
  pages     = {2350--2354},
  year      = {2019},
  doi       = {10.21437/Interspeech.2019-2219},
  eprint    = {1812.08466},
  archivePrefix = {arXiv}
}
@inproceedings{gui2024fad,
  author    = {Gui, Azalea and Gamper, Hannes and Braun, Sebastian and Emmanouilidou, Dimitra},
  title     = {Adapting {Frechet} Audio Distance for Generative Music Evaluation},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1331--1335},
  year      = {2024},
  doi       = {10.1109/ICASSP48485.2024.10446663},
  eprint    = {2311.01616},
  archivePrefix = {arXiv}
}
@misc{chung2025kad,
  author = {Chung, Yoonjin and Eu, Pilsun and Lee, Junwon and Choi, Keunwoo and Nam, Juhan and Chon, Ben Sangbae},
  title  = {{KAD}: No More {FAD}! An Effective and Efficient Evaluation Metric for Audio Generation},
  year   = {2025},
  eprint = {2502.15602},
  archivePrefix = {arXiv}
}
@misc{tjandra2025audiobox,
  author = {Tjandra, Andros and Wu, Yi-Chiao and Guo, Baishan and Hoffman, John and Ellis, Brian and Vyas, Apoorv and Shi, Bowen and Chen, Sanyuan and Le, Matt and Zacharov, Nick and Wood, Carleigh and Lee, Ann and Hsu, Wei-Ning},
  title  = {Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound},
  year   = {2025},
  eprint = {2502.05139},
  archivePrefix = {arXiv}
}
@inproceedings{chinen2020visqol,
  author    = {Chinen, Michael and Lim, Felicia S. C. and Skoglund, Jan and Gureev, Nikita and O'Gorman, Feargus and Hines, Andrew},
  title     = {{ViSQOL} v3: An Open Source Production Ready Objective Speech and Audio Metric},
  booktitle = {International Conference on Quality of Multimedia Experience (QoMEX)},
  year      = {2020},
  doi       = {10.1109/QoMEX48832.2020.9123150},
  eprint    = {2004.09584},
  archivePrefix = {arXiv}
}
@inproceedings{wu2023clap,
  author    = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  title     = {Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10095969},
  eprint    = {2211.06687},
  archivePrefix = {arXiv}
}
@inproceedings{vinay2022evaluating,
  author    = {Vinay, Ashvala and Lerch, Alexander},
  title     = {Evaluating Generative Audio Systems and Their Metrics},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2022},
  eprint    = {2209.00130},
  archivePrefix = {arXiv}
}
@inproceedings{grotschla2025benchmarking,
  author    = {Gr{\"o}tschla, Florian and Solak, Ahmet and Lanzend{\"o}rfer, Luca A. and Wattenhofer, Roger},
  title     = {Benchmarking Music Generation Models and Metrics via Human Preference Studies},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10887745},
  eprint    = {2506.19085},
  archivePrefix = {arXiv}
}
@article{schoeffler2018webmushra,
  author  = {Schoeffler, Michael and Bartoschek, Sarah and St{\"o}ter, Fabian-Robert and Roess, Marlene and Westphal, Susanne and Edler, Bernd and Herre, J{\"u}rgen},
  title   = {{webMUSHRA}: A Comprehensive Framework for Web-based Listening Tests},
  journal = {Journal of Open Research Software},
  volume  = {6},
  number  = {1},
  pages   = {8},
  year    = {2018},
  doi     = {10.5334/jors.187}
}
@inproceedings{jillings2015waet,
  author    = {Jillings, Nicholas and De Man, Brecht and Moffat, David and Reiss, Joshua D.},
  title     = {Web Audio Evaluation Tool: A Browser-Based Listening Test Environment},
  booktitle = {Proc. 12th Sound and Music Computing Conference (SMC)},
  year      = {2015},
  doi       = {10.5281/zenodo.851157}
}
@inproceedings{deman2014ape,
  author    = {De Man, Brecht and Reiss, Joshua D.},
  title     = {{APE}: Audio Perceptual Evaluation Toolbox for {MATLAB}},
  booktitle = {Audio Engineering Society Convention 136},
  year      = {2014},
  note      = {Convention number not re-verified; title, authors and date from OpenAlex}
}
@inproceedings{steinmetz2021pyloudnorm,
  author    = {Steinmetz, Christian J. and Reiss, Joshua D.},
  title     = {pyloudnorm: A Simple Yet Flexible Loudness Meter in {Python}},
  booktitle = {Audio Engineering Society Convention 150},
  year      = {2021},
  note      = {Convention number not re-verified; title, authors and date from OpenAlex (QMRO handle 123456789/80278)}
}
@techreport{itu1770,
  author      = {{ITU-R}},
  title       = {Recommendation {ITU-R BS.1770-5}: Algorithms to Measure Audio Programme Loudness and True-Peak Audio Level},
  institution = {International Telecommunication Union},
  year        = {2023}
}
@techreport{itu1534,
  author      = {{ITU-R}},
  title       = {Recommendation {ITU-R BS.1534-3}: Method for the Subjective Assessment of Intermediate Quality Level of Audio Systems},
  institution = {International Telecommunication Union},
  year        = {2015}
}
@techreport{itu1387,
  author      = {{ITU-R}},
  title       = {Recommendation {ITU-R BS.1387-2}: Method for Objective Measurements of Perceived Audio Quality},
  institution = {International Telecommunication Union},
  year        = {2023}
}
@techreport{ebur128,
  author      = {{European Broadcasting Union}},
  title       = {{EBU R 128}: Loudness Normalisation and Permitted Maximum Level of Audio Signals},
  institution = {EBU},
  year        = {2023},
  note        = {Version 5.0}
}

% ---------- B/C. Software and datasets ----------
@misc{matchering,
  author = {Grishakov, S. and Yu, C.-Y. and Zicklag},
  title  = {Matchering 2.0: Audio Matching and Mastering {Python} Library},
  year   = {2022},
  howpublished = {\url{https://github.com/sergree/matchering}},
  note   = {Version 2.0.6, GPL-3.0. Author name as cited in the ITO-Master paper}
}
@misc{rafii2017musdb18,
  author = {Rafii, Zafar and Liutkus, Antoine and St{\"o}ter, Fabian-Robert and Mimilakis, Stylianos Ioannis and Bittner, Rachel},
  title  = {{MUSDB18}: A Corpus for Music Separation},
  year   = {2017},
  doi    = {10.5281/zenodo.1117372}
}
@misc{rafii2019musdb18hq,
  author = {Rafii, Zafar and Liutkus, Antoine and St{\"o}ter, Fabian-Robert and Mimilakis, Stylianos Ioannis and Bittner, Rachel},
  title  = {{MUSDB18-HQ}: An Uncompressed Version of {MUSDB18}},
  year   = {2019},
  doi    = {10.5281/zenodo.3338373}
}
@inproceedings{pereira2023moisesdb,
  author    = {Pereira, Igor and Ara{\'u}jo, Felipe and Korzeniowski, Filip and Vogl, Richard},
  title     = {{MoisesDB}: A Dataset for Source Separation Beyond 4-Stems},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2023},
  eprint    = {2307.15913},
  archivePrefix = {arXiv}
}
@inproceedings{bittner2014medleydb,
  author    = {Bittner, Rachel M. and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan Pablo},
  title     = {{MedleyDB}: A Multitrack Dataset for Annotation-Intensive {MIR} Research},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2014}
}
@inproceedings{manilow2019slakh,
  author    = {Manilow, Ethan and Wichern, Gordon and Seetharaman, Prem and Le Roux, Jonathan},
  title     = {Cutting Music Source Separation Some {Slakh}: A Dataset to Study the Impact of Training Data Quality and Quantity},
  booktitle = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  pages     = {45--49},
  year      = {2019},
  doi       = {10.1109/WASPAA.2019.8937170}
}
@inproceedings{defferrard2017fma,
  author    = {Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  title     = {{FMA}: A Dataset For Music Analysis},
  booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2017},
  eprint    = {1612.01840},
  archivePrefix = {arXiv}
}
@inproceedings{bogdanov2019mtgjamendo,
  author    = {Bogdanov, Dmitry and Won, Minz and Tovstogan, Philip and Porter, Alastair and Serra, Xavier},
  title     = {The {MTG-Jamendo} Dataset for Automatic Music Tagging},
  booktitle = {Machine Learning for Music Discovery Workshop, ICML},
  year      = {2019}
}
@misc{roy2025jamendomaxcaps,
  author = {Roy, Abhinaba and Liu, Renhang and Lu, Tongyu and Herremans, Dorien},
  title  = {{JamendoMaxCaps}: A Large Scale Music-Caption Dataset with Imputed Metadata},
  year   = {2025},
  eprint = {2502.07461},
  archivePrefix = {arXiv}
}
@inproceedings{manco2023songdescriber,
  author    = {Manco, Ilaria and Weck, Benno and Doh, SeungHeon and Won, Minz and Zhang, Yixiao and Bogdanov, Dmitry and Wu, Yusong and Chen, Ke and Tovstogan, Philip and Benetos, Emmanouil and Quinton, Elio and Fazekas, Gy{\"o}rgy and Nam, Juhan},
  title     = {The Song Describer Dataset: A Corpus of Audio Captions for Music-and-Language Evaluation},
  booktitle = {NeurIPS Workshop on Machine Learning for Audio},
  year      = {2023},
  eprint    = {2311.10057},
  archivePrefix = {arXiv}
}
@inproceedings{deman2014omt,
  author    = {De Man, Brecht and Mora-Mcginity, Mariano and Fazekas, Gy{\"o}rgy and Reiss, Joshua D.},
  title     = {The Open Multitrack Testbed},
  booktitle = {Audio Engineering Society Convention 137},
  year      = {2014},
  note      = {Engineering Brief 165}
}
@inproceedings{petermann2022dnr,
  author    = {Petermann, Darius and Wichern, Gordon and Wang, Zhong-Qiu and Le Roux, Jonathan},
  title     = {The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2022},
  pages     = {526--530},
  doi       = {10.1109/ICASSP43922.2022.9746005},
  eprint    = {2110.09958},
  archivePrefix = {arXiv}
}
```
