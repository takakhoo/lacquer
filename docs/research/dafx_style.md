# How DAFx papers are written: conventions, author guidelines, and a reading of 17 recent papers

Compiled 2 October 2026 for the remaster paper (hybrid DSP plus learned restoration and mastering of finished stereo mixes), targeting DAFx27 in Cremona. Everything below comes from the DAFx proceedings archive, the conference sites, the official LaTeX template, and a full read of 13 papers (plus a skim of 4 more). Anything inferred rather than stated by a source is marked "inferred".

Scope note. DAFx27 (30th edition) is at the Cremona Campus of Politecnico di Milano, 24 to 27 August 2027, site https://dafx27.deib.polimi.it/, contact dafx27@gmail.com. As of today that site lists topics only; no call for papers, page limit, template, or deadlines are published yet. The DAFx26 (MIT, Cambridge MA, 1 to 4 September 2026) guidelines are the latest available and are what this document reports. The template has been handed from one local committee to the next every year since DAFx-00, so expect DAFx27 to differ only in the year, venue strings, and possibly the blind/print switch.

## 1. Papers studied

Verified PDF locations. The dafx.de archive serves every paper directly, with a year-dependent path:

- DAFx23: `https://www.dafx.de/paper-archive/2023/DAFx23_paper_NN.pdf`
- DAFx24: `https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_NN.pdf` (also mirrored at `https://github.com/IoSR-Surrey/DAFx24-Proceedings/raw/main/papers/DAFx24_paper_NN.pdf`)
- DAFx25: `https://www.dafx.de/paper-archive/2025/DAFx25_paper_NN.pdf`
- DAFx26: `https://www.dafx.de/paper-archive/2026/papers/DAFx26_paper_NN.pdf`

The archive has a search API (`https://www.dafx.de/paper-archive/api/v1/search?q=...&years[]=2025`) that returns title, authors, abstract and filename as JSON; this is how the list below was verified. Full proceedings PDFs: DAFx23 https://dafx23.create.aau.dk/wp-content/uploads/2023/09/DAFX23_Proceedings.pdf (387 pp.), DAFx24 https://github.com/IoSR-Surrey/DAFx24-Proceedings/raw/main/DAFx24Proceedings.pdf, DAFx25 https://dafx25.dii.univpm.it/wp-content/uploads/2025/09/DAFx25Proceedings.pdf. Page ranges below were read from the printed footers of each PDF.

Award sources: DAFx23 awards page https://dafx23.create.aau.dk/index.php/awards/; DAFx24 award certificates (images) on https://dafx24.surrey.ac.uk/ (the site states "a three-way-tie for Best Paper given the top review scores"); DAFx25 awards reported by Edinburgh AAG https://www.acoustics.ed.ac.uk/?p=2054 and Aalto https://www.aalto.fi/en/news/double-triumph-aalto-acoustics-researchers-win-big-in-italy; DAFx26 best paper reported by FAU LMS https://www.lms.tf.fau.eu/best-paper-award-at-dafx26/.

### Read in full (13)

1. G. Dal Santo, K. Prawda, S. J. Schlecht, and V. Välimäki, "Differentiable feedback delay network for colorless reverberation," in Proc. 26th Int. Conf. Digital Audio Effects (DAFx23), Copenhagen, Denmark, Sept. 2023, pp. 244-251. DAFx23 Best Paper. PDF: https://www.dafx.de/paper-archive/2023/DAFx23_paper_32.pdf
   What it does: optimizes the feedback matrix and I/O gains of a frequency-sampled differentiable FDN with a spectral flatness loss plus a temporal sparsity loss, then validates with modal analysis and a MUSHRA test. The model listening-test report in this set.

2. P. Meier, S. Schwär, and M. Müller, "A real-time approach for estimating pulse tracking parameters for beat-synchronous audio effects," in Proc. 27th Int. Conf. Digital Audio Effects (DAFx24), Guildford, UK, Sept. 2024, pp. 314-321. DAFx24 Best Paper (one of three). PDF: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_23.pdf
   What it does: normalizes a real-time predominant-local-pulse buffer into LFO and confidence control signals, demonstrated in a JUCE plugin with three mixing case studies. No listening test; evaluation is by case study.

3. A. Carson, A. Wright, J. Chowdhury, V. Välimäki, and S. Bilbao, "Sample rate independent recurrent neural networks for audio effects processing," in Proc. DAFx24, pp. 17-24. DAFx24 Best Presentation. PDF: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_68.pdf
   What it does: compares four ways to run a pre-trained RNN effect model at a different sample rate, with a linear one-pole analysis, then SNR and aliasing metrics on 18 GuitarML models.

4. E. Moliner, M. Turunen, F. Elvander, and V. Välimäki, "A diffusion-based generative equalizer for music restoration," in Proc. DAFx24, pp. 25-32. PDF: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_71.pdf
   What it does: BABE-2, blind restoration of 78 rpm piano and vocal recordings by jointly estimating a piecewise-linear zero-phase EQ curve and sampling a diffusion prior. Closest in spirit to the remaster paper (restoration of finished recordings, objective FAD and LTAS metrics, qualitative spectrogram analysis).

5. C.-Y. Yu, C. Mitcheltree, A. Carson, S. Bilbao, J. D. Reiss, and G. Fazekas, "Differentiable all-pole filters for time-varying audio systems," in Proc. DAFx24, pp. 345-352. PDF: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_75.pdf
   What it does: exact backprop through time-varying all-pole filters (torchlpc), applied to a phaser, a TB-303 style synth, and a feed-forward compressor fitted to an LA-2A (torchcomp). Relevant for any differentiable compressor or limiter in a hybrid system.

6. E. Moliner, M. Švento, A. Wright, L. Juvela, P. Rajmic, and V. Välimäki, "Unsupervised estimation of nonlinear audio effects: Comparing diffusion-based and adversarial approaches," in Proc. 28th Int. Conf. Digital Audio Effects (DAFx25), Ancona, Italy, Sept. 2025, pp. 366-373. DAFx25 Second Best Paper. PDF: https://www.dafx.de/paper-archive/2025/DAFx25_paper_75.pdf
   What it does: blind effect estimation from unpaired data via diffusion EM versus an adversarial baseline, with black-box (GCN, S4) and gray-box (Wiener-Hammerstein) operators, swept over data quantity.

7. V. Välimäki, S. Bilbao, S. J. Schlecht, R. Salmi, and D. Zicarelli, "Zero-phase sound via giant FFT," in Proc. DAFx25, pp. 290-297. DAFx25 Third Best Paper. PDF: https://www.dafx.de/paper-archive/2025/DAFx25_paper_34.pdf
   What it does: a pure DSP creative effect paper (whole-song FFT, phase zeroing, fades, gain compensation, stereo variant, Max/MSP real-time version). Shows how DAFx rewards clear signal analysis with audio examples and no learning at all.

8. P. Sarkar and P. Lindborg, "Neural-driven multi-band processing for automatic equalization and style transfer," in Proc. DAFx25, pp. 382-389. PDF: https://www.dafx.de/paper-archive/2025/DAFx25_paper_81.pdf
   What it does: a TCN predicts parameters of a six-band differentiable PEQ plus per-band compressor (dasp-pytorch), trained self-supervised on MUSDB18; objective metrics (SI-SDR, PESQ, LUFS difference, STFT loss) and a 20-participant MOS test. The nearest "automatic mastering-style" paper in this set.

9. X. He, D. Williams, and B. Fazenda, "Evaluating the performance of objective audio quality metrics in response to common audio degradations," in Proc. DAFx25, pp. 235-242. PDF: https://www.dafx.de/paper-archive/2025/DAFx25_paper_5.pdf
   What it does: tests PEAQ, PEMO-Q, ViSQOL, HAAQI on hum, hiss, clipping and glitches added to MUSDB18 mixes at -14 LUFS. Useful precedent for choosing and justifying objective metrics on full mixes.

10. G. Dal Santo, X. Pi, K. Prawda, S. J. Schlecht, and V. Välimäki, "Shimmer reverberation with nonlinear feedback delay networks," in Proc. 29th Int. Conf. Digital Audio Effects (DAFx26), Cambridge, MA, USA, Sept. 2026, pp. 40-47. DAFx26 Best Paper. PDF: https://www.dafx.de/paper-archive/2026/papers/DAFx26_paper_05.pdf
    What it does: five nonlinear or time-varying operations inside an FDN loop, energy-ratio histograms over 336 minutes of stems, spectrogram analysis, design guidance, audio examples; no listening test.

11. F. Fontana, S. Pasin, A. Bernardini, and S. D'Angelo, "A clipping prevention method for all-pass digital filters with time-varying coefficients," in Proc. DAFx26, pp. 364-371. PDF: https://www.dafx.de/paper-archive/2026/papers/DAFx26_paper_45.pdf
    What it does: algebraic bound on coefficient increments so a time-varying all-pass never exceeds a clip threshold; case studies on sines and a music excerpt, timing table in MATLAB and C++. Directly relevant to limiter and clipping discussions.

12. B. R. Thompson and M. C. Heilemann, "Evaluating dynamic range compressor models using control-voltage measurements: An approach and dataset," in Proc. DAFx26, pp. 241-248. PDF: https://www.dafx.de/paper-archive/2026/papers/DAFx26_paper_30.pdf
    What it does: argues waveform L1 and MSTE are poor proxies for gain-trajectory error, trains torchcomp under three losses, releases an SSL bus compressor dataset with measured gain-reduction CV. Relevant to how you evaluate a learned or hybrid dynamics stage.

13. Q. Yang, T. Berg-Kirkpatrick, J. McAuley, and Z. Novack, "WildFX: A DAW-powered pipeline for in-the-wild audio FX graph modeling," in Proc. DAFx26, pp. 453-460. PDF: https://www.dafx.de/paper-archive/2026/papers/DAFx26_paper_56.pdf
    What it does: Dockerized REAPER pipeline for graph-labelled mixing datasets; the one paper in the set with an explicit "Limitations & Discussions" section.

### Skimmed for citation and structure only (4)

14. O. Massi, E. Manino, and A. Bernardini, "Wave digital modeling of circuits with multiple one-port nonlinearities based on Lipschitz-bounded neural networks," in Proc. DAFx24, pp. 119-126. DAFx24 Best Paper (one of three). PDF: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_45.pdf
15. G. Lee, H. Kim, J. Lee, and J. Nam, "CONMOD: Controllable neural frame-based modulation effects," in Proc. DAFx24, pp. 9-16. DAFx24 Best Paper (one of three). PDF: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_61.pdf
16. R. Simionato and S. Fasciani, "Fully conditioned and low-latency black-box modeling of analog compression," in Proc. DAFx23, pp. 287-294. PDF: https://www.dafx.de/paper-archive/2023/DAFx23_paper_10.pdf
17. Y. Gu, R. Zhang, L. Juvela, and Z. Wu, "Solid State Bus-Comp: A large-scale and diverse dataset for dynamic range compressor virtual analog modeling," in Proc. DAFx25, pp. 55-62. PDF: https://www.dafx.de/paper-archive/2025/DAFx25_paper_13.pdf

Other award winners, for completeness: DAFx23 Best Poster went to Giampiccolo, D'Angelo, Bernardini and Sarti, "A quadric surface model of vacuum tubes for virtual analog applications" (pp. 296-303); DAFx24 Best Poster to Sun and Depalle, "Hybrid audio inpainting approach with structured sparse decomposition and sinusoidal modeling"; DAFx25 First Best Paper to Risse, Hélie and Bilbao, "Power-balanced drift regulation for scalar auxiliary variable methods" and joint Third to Zheleznov, Bilbao, Wright and King, "Learning nonlinear dynamics in physical modelling synthesis using neural ordinary differential equations" (DAFx25_paper_37.pdf).

Every one of the 17 PDFs is exactly 8 pages.

## 2. How DAFx papers are written

### 2.1 Section structure and names

The template fixes the skeleton: a centred all-caps title, authors with affiliation blocks, ABSTRACT, then numbered all-caps sections, numbered subsections (2.1, 2.1.1), ACKNOWLEDGMENTS, REFERENCES. In the 17 papers the sequence is almost always one of these two:

- Introduction; Background (or Related Work, or Problem Statement); Method (named for the thing: "FDN Optimization", "Proposed Methodology", "BABE-2: Unique Contributions", "Unsupervised Operator Estimation"); Experiments / Evaluation (sometimes split into Objective Evaluation and Perceptual Evaluation); Results (or Results and Analysis); Discussion (optional); Conclusion(s); Acknowledgments; References.
- For DSP-analysis papers: Introduction; Method; Case Study / Analysis; Discussion; Conclusions; References (Fontana et al., Välimäki et al.).

Observed counts: 5 to 8 numbered body sections before Acknowledgments. "Conclusions" (plural) and "Conclusion" both appear; "Conclusions and Further Work" and "Conclusion and Future Work" also appear. A separate "Discussion" section appears in about a third (Moliner 24 "Discussion: Restoring Historical Voices", Fontana "Discussion", Sarkar "6.4 Discussion", WildFX "Limitations & Discussions"). Appendices appear after References in two papers (He et al. "Appendix: Effectiveness Test Details"; the template itself ends with an "Appendix: Margin Check").

### 2.2 Abstract: length and shape

Single paragraph, 130 to 200 words (Dal Santo 158, Moliner 25 143, Carson about 170, Sarkar about 200). Shape, almost without exception:

1. One or two sentences of context or problem, stated as a fact about the field. Examples: "Artificial reverberation algorithms often suffer from spectral coloration" (Dal Santo et al., DAFx23); "Accurately estimating nonlinear audio effects without access to paired input-output signals remains a challenging problem" (Moliner et al., DAFx25); "Infinite impulse response filters are an essential building block of many time-varying audio systems" (Yu et al., DAFx24).
2. "This paper proposes / introduces ..." or "We introduce / investigate ..." naming the method and the object.
3. One or two sentences of mechanism (what is optimized, what loss, what prior).
4. One or two sentences of evidence, often with a concrete claim: "the diffusion-based approach provides more stable results and is less sensitive to data availability" (Moliner et al., DAFx25); He et al. even give standard deviations in the abstract.
5. A closing sentence that states significance or points to artefacts: "We make our code and audio samples available and provide the trained audio effect and synth models in a VST plugin" (Yu et al.); "The code is available at: https://github.com/..." (Yang et al.); "To our knowledge, this work provides the first in-depth study into this problem" (Carson et al.).

No abstract in the set mentions limitations. Hedging is light ("to the best of our knowledge" appears when a novelty claim is made).

### 2.3 How the introduction opens

Pattern across all 13 full reads: paragraph 1 states what the effect or problem is and why practitioners care, in plain declarative sentences, usually with one or two foundational citations in the first two sentences. Nobody opens with a scenario, a quotation (He et al. is the one exception, quoting a mixing textbook), or a definition set off from the prose. Examples of openings: "Historical music recordings suffer from severe impairment due to limitations of the physical recording media" (Moliner 24); "Many audio effects and instruments, such as echo, flanger, tremolo, or synthesizers, are often strongly aligned with the rhythmic structure of the music" (Meier 24); "A dynamic range compressor (DRC) is a processor that applies a time-varying attenuation to a signal as a function of its level" (Thompson 26).

Then, typically: two to four paragraphs of prior work woven into the narrative with bracketed numeric citations (not a separate related-work survey), a gap sentence ("However, ..."), and a contribution paragraph that starts "In this paper, we ..." or "In this study, we present ...". Yu et al. use an explicit numbered list: "Our contributions are threefold". Nearly every paper ends the introduction with a roadmap paragraph: "The paper is organized as follows. Section 2 ... Section 6 offers concluding remarks" (Dal Santo); "This paper is structured as follows. Sec. 2 outlines ..." (Moliner 24); "The remainder of this paper is organized as follows" (Sarkar, Dal Santo 26, Thompson). Only Yu et al. and He et al. skip or shorten it. Introduction length is about 0.5 to 0.8 of a page.

### 2.4 Use of "we"

First person plural is the default. Counts of "we" in the body (excluding references): Dal Santo 28, Moliner 24 65, Yu 83, Carson 41, Meier 78, Moliner 25 57, Sarkar 40, Dal Santo 26 about 40, WildFX 29. Three papers are written mostly in the passive or with "this study" (Fontana 4, He 3, Thompson 10). "We" is used for choices and claims ("we propose", "we found", "we recommend operating at double precision", "we hypothesize"), while results sentences often shift to the impersonal ("It can be seen that", "The results show that"). "I" never appears. "The authors" appears only in acknowledgments and self-citation phrases ("the authors' prior research").

### 2.5 How equations are introduced

The template instruction is literal: "Equations should be placed on separate lines and numbered". In practice:

- A lead-in clause ends with "is", "as", "given by", or a colon, then the displayed equation, then "where" defines every new symbol in one sentence: "The transfer function of the FDN is [eq. (1)] where A is the N × N feedback matrix, N being the number of delay lines" (Dal Santo 23).
- In-text references are "(1)", "Eq. (4)", "Eqs. (12) and (13)", "equation 2"; the first two forms dominate, and the Aalto papers use "(1)" bare after first mention.
- Notation conventions are stated once when they matter: "we denote discrete-time dependence in signals using square brackets, and in filter coefficients using round brackets" (Fontana 26).
- Algorithms are given as numbered pseudocode floats ("Algorithm 1 Inference phase of the BABE-2 method"; Fontana gives two).
- Loss functions and metrics are always written out explicitly with their normalization (SNR, ESR, MSS, FAD, LTAS distance), even when standard.

### 2.6 Figures and tables: captions and conventions

- Figures: caption below, "Figure N:" in the caption, referred to in text as "Fig. N" (Aalto, Edinburgh, Polimi papers) or "Figure N" (KAIST, Erlangen, UCSD papers); both are accepted. Captions are full sentences and frequently state what the reader should see: "Results of the listening test on exponential decaying IRs (expDE), showing that the proposed DiffFDN has the highest median score of colorlessness in all cases" (Dal Santo 23); "Both zero-phase spectra are remarkably similar to the original" (Välimäki 25). Multi-panel figures are labelled (a), (b), (c) and each panel is explained in the caption. Box-plot conventions (median mark, 25th and 75th percentiles, whiskers, outliers, notches at 5% significance) are spelled out in prose the first time.
- Typical figure set in an 8-page paper: 4 to 9 figures, including one block diagram of the system (almost universal, usually Fig. 1 or 2), spectrograms or magnitude responses, and one results plot. Figures spanning two columns are common for results grids.
- Tables: caption above (template rule), booktabs rules, 1 to 5 tables. Captions name the content and the reading direction: "Objective evaluation on historical recordings. The best results of each column for each experiment are bolded" (Moliner 24); "Lower is better for all metrics" (Yang 26); arrows (FAD ↓) in headers (Moliner 24, Moliner 25). Tables are also used for parameter ranges and hyperparameters (Yu Table 1; Sarkar Tables 1 and 2; Thompson Table 2 lists hardware control settings).
- Hardware and timing are reported concretely: "M1 Pro MacBook for one optimisation step (forward + backward, one thread, single precision)" (Yu); "NVIDIA H200 GPU" (Moliner 25); MATLAB and gcc -O3 timings in a table (Fontana).

### 2.7 How much related work

Short. Either no separate section (prior work covered in 2 to 4 introduction paragraphs: Dal Santo 23, Carson, Meier, Fontana, Thompson) or a "Related Work(s)" / "Background" section of 0.3 to 0.6 page (Yu, Sarkar, WildFX, Moliner 24 "Background" which is technical background on diffusion posterior sampling rather than a survey). No paper has a related-work section longer than about two-thirds of a page. The related work is used to position one specific gap, and DAFx self-citation is heavy: the papers cite 3 to 10 earlier DAFx papers each.

### 2.8 Listening tests: whether and how they are reported

Only 3 of 17 papers report a formal listening test; the rest rely on objective metrics, analysis plots, and audio examples on a companion web page (which is close to universal: "Audio examples are available at the companion page", footnoted URL on page 1 or in the abstract, often research.spa.aalto.fi/publications/papers/dafxNN-topic/ or a github.io page). Several papers explicitly defer: "For future research, we aim to conduct subjective testing to validate the case studies described in this paper" (Meier 24); Moliner 24 settles on "a qualitative analysis" with annotated spectrograms.

When a listening test is run, the reporting template (Dal Santo 23, the Best Paper) is: standard named (MUSHRA, ITU-R BS.1534-3) and software (webMUSHRA); number of pages and stimuli per page; training page and loudness setting; reference and anchor defined and justified; venue and headphones ("sound-insulated booth at the Aalto Acoustics Lab, with participants wearing Sennheiser HD650 headphones"); participants ("12 listeners. One participant was excluded ... average age ... 28.6 years with standard deviation of 4.1, and none of them reported any hearing impairments"; expertise stated); slider labels; statistics (Shapiro-Wilk normality test, then Wilcoxon signed-rank with Bonferroni correction, p-values quoted for the non-significant pair); results as box plots with medians quoted in text (50.5, 74, 77). Sarkar 25 is lighter: "20 participants (mean age 26) using a 5-point MOS scale", Shapiro-Wilk, paired t-tests, significance stars in the table. Lee et al. (DAFx24 distortion recovery, skimmed) also report participants and listeners. Inferred: a listening test is welcome and strengthens a restoration or mastering paper, but reviewers accept objective evaluation plus audio examples when the test is well justified or deferred honestly.

### 2.9 Limitations and future work

Limitations are stated plainly, usually in the Discussion or in the Conclusions paragraph, and they are specific and technical rather than generic: "In early experiments we observed that the TD implementation could become unstable during training ... we recommend operating at double precision" (Yu); "in some cases it fails and produces noisy, highly-aliased, inaccurate output" (Carson on APDL); "BABE-2 hallucinates consonants at statistically plausible locations, but these may not correspond to the lyrics" (Moliner 24); "A key limitation of the diffusion-based method is the need for a separately pre-trained model on clean guitar signals" (Moliner 25). Reproducibility caveats are stated outright: "each setting is trained once. The results should therefore be interpreted as an initial validation ... rather than a statistically conclusive comparison" (Yang 26). WildFX is the only paper with a section titled "Limitations & Discussions". Future work is one to four sentences at the end of the conclusion ("Future work includes designing an adaptive attenuation filter ..."; "Our future work involves extending the backpropagation algorithm ...").

### 2.10 Conclusion length

Usually 100 to 350 words, one to three paragraphs: restate what was done, the one or two headline results (sometimes with numbers), then limitations or future work. Moliner 24 is 100 words; Dal Santo 23 about 190; Sarkar about 250 plus a future-work paragraph; Thompson about 230. Outliers: Välimäki 25 has a long reflective conclusion (about 590 words) that doubles as a discussion. Nobody adds new results in the conclusion.

### 2.11 References

13 to 50 entries, median about 29 (counts: 34, 24, 16, 32, 26, 22, 35, 30, 48, 23, 29, 50, 26, 15, 29, 19, 43, 33, 41). IEEE numeric style, numbered in order of first appearance, produced by the supplied IEEEtranDAFx.bst. DAFx papers cite earlier DAFx papers constantly, plus JAES, IEEE/ACM TASLP, ICASSP, and the Zölzer DAFX book. arXiv preprints are cited when needed. Dataset, software and plugin references (GitHub, JUCE, REAPER, dasp-pytorch) are given as numbered references too.

### 2.12 Tone

Formal but plain. Direct technical sentences, little hedging, no marketing adjectives. Occasional light touches survive review: "a common bane of systems utilizing comb filters is sound coloration" (Dal Santo 23); "Rather surprisingly, a recent paper showed ..." and "This teaches us that simply zeroing the phase does not retain the spectral content" (Välimäki 25). Claims are quantified ("up to 50 dB increase ... in SNR", "two to three times faster than FS", "the largest gain-reduction error is 85.9% greater than the smallest"). "Novel" appears in most abstracts or introductions and does not seem to be penalized. Spelling is mixed British and American across papers (the template says to follow IEEE style, which is American); each paper is internally consistent.

### 2.13 Other recurring features

- Precise training and implementation details in prose: optimizer, learning rate, batch size, epochs or steps, segment length, sample rate, FFT sizes, window types, loss weights (every learned-model paper does this, usually in an "Experimental Details" or "Implementation Details" subsection).
- Baselines are named and ablations are labelled in tables ("BABE-2 w/o noise reg.").
- A supervised upper bound is reported when the method is unsupervised (Moliner 25).
- Companion web page with audio, often using the trackswitch.js player, and a GitHub repository; VST or Max/MSP implementations are mentioned when they exist and are treated as evidence of practicality.
- Acknowledgments are short and factual: funder and grant number, research visits with dates, compute resources, proofreading thanks.
- First page carries the copyright notice (auto-generated by the style) at the bottom of the left column, below any \thanks footnotes.

## 3. Author guidelines (DAFx26 edition, the latest published)

URLs:
- Information for authors (templates, review flow): https://dafx26.mit.edu/authors/
- Call for papers: https://dafx26.mit.edu/call-for-papers
- LaTeX template: https://dafx26.mit.edu/assets/DAFx26_LaTeX_v3.zip (contains DAFx26_tmpl_v3.tex, dafx26v3.sty, IEEEtranDAFx.bst, DAFx26_tmpl.bib, two example figure PDFs). Demo template DAFx26_LaTeX_DEMO.zip and challenge template DAFx26_LaTeX_CHALLENGE.zip at the same path.
- Review decision flowchart: https://dafx26.mit.edu/assets/decision.png
- Proceedings archive and ISSN: https://dafx.de/paper-archive/ (ISSN 2413-6689 online, 2413-6700 print)
- Older generic template page (DAFx-02 era, obsolete but still linked from dafx.de): https://dafx.de/conference_templates.html
- Previous CFPs with the same policies: https://dafx24.surrey.ac.uk/call-for-papers/ and https://dafx25.dii.univpm.it/call-for-papers/
- DAFx27 site (no CFP yet as of 2 Oct 2026): https://dafx27.deib.polimi.it/

Facts:

- Page limit. "Prospective authors are invited to submit full-length papers (8 pages maximum)" (DAFx26 CFP); the template's conclusion repeats "max. 8 pages both oral and poster presentations", and the template says the last page "at most, will have to be DAFx.8". DAFx24 and DAFx25 CFPs: "eight pages maximum, for both oral and poster presentations". Do references count? No source says references are excluded, the page-numbering remark counts the whole PDF, and all 17 papers checked are exactly 8 pages with references inside. Inferred: references count toward the 8 pages.
- Format. A4, two columns, text area 175 mm x 226 mm, Times, 10 pt default with \ninept giving 9 pt (every paper read uses the 9 pt setting). Leave page numbers as DAFx.1 and so on; proceedings numbers are added in post-processing. Compile with pdflatex; figures as PDF, PNG or JPG.
- Anonymization. "This year DAFx reviews will be double-blind. For the initial submission, please make sure to remove all identifying information (e.g., author names, affiliations, etc) from your paper and any supplemental material" (DAFx26 CFP and template). Mechanism: `\usepackage[blind]{dafx26v3}` for submission, `\usepackage[print]{dafx26v3}` for camera-ready. Blind mode replaces the author block with "Anonymous Authors", "Dept. of Anonymous Studies", and prints the copyright as "© 2026 the Anonymous Authors"; the funding footnote becomes "Thanks to the anonymous supporters". DAFx24 and DAFx25 were single-blind as far as the CFPs state (uncertain; the earlier CFPs do not mention anonymization). Also anonymize the companion web page and repository for submission.
- Preprint policy. DAFx26 CFP: "We discourage authors from posting preprints (i.e., on arXiv)." If one is posted, authors should "clearly identify it as a non-peer-reviewed, non-accepted document." (Authors do still post: Moliner 24 cites its own arXiv appendix.) DAFx24 and DAFx25 CFPs have no preprint statement.
- Copyright notice line. Generated by the style file from \paperauthorA; do not type it. Camera-ready text: "Copyright: © 2026 <First Author> et al. This is an open-access article distributed under the terms of the Creative Commons Attribution 4.0 International License, which permits unrestricted use, distribution, adaptation, and reproduction in any medium, provided the original author and source are credited." ("et al." is dropped for a single author.) Proceedings are CC BY 4.0, free on dafx.de after the conference, and indexed in Scopus ("Volumes 1998 to 2024 of DAFx proceedings are now indexed in Scopus and this will apply similarly to DAFx25 proceedings", DAFx25 CFP).
- Figures and tables. "All figures should be centered on the column (or page, if the figure spans both columns). Figure captions should follow each figure"; "Short captions are centred, long captions (more than 1 line) are justified"; vector graphics preferred; bitmaps at 300 DPI or higher; figure text no smaller than footnote size; "Table captions should precede each table"; the template loads booktabs and subfig. SI units via siunitx. Equations on separate numbered lines.
- Style authority. "DAFx follows IEEE guidelines for writing style, spelling, grammar, citation format, and most other style conventions" (template abstract). References "will be numbered in order of appearance"; "Use the standard IEEE reference style"; BibTeX with IEEEtranDAFx recommended.
- Required or expected sections. The template's own skeleton is Abstract, Introduction, (body), Conclusions, Acknowledgments, References. Acknowledgments is where the funding line goes ("Many thanks to the great number of anonymous reviewers!" is the placeholder); per-author funding can also go in a title-page footnote via `\sthanks{This work was supported by the XYZ Foundation}`. Affiliation headers exist for 1 to 4 distinct affiliations (`\affiliation`, `\twoaffiliations`, `\threeaffiliations`, `\fouraffiliations`), with department, institution, city and e-mail per block. Word template only "by request" from the organizers.
- Submission. PDF through EasyChair; "Submitted papers must be camera-ready and conform to the format specified in the template". DAFx26 dates: submission 30 March 2026, edits until 3 April, notification 15 May, borderline revision and rebuttal 5 June, camera-ready 3 July. DAFx25 ran on the same rhythm (31 March, 6 May, 30 May, 9 June, 1 July). Expect DAFx27 deadlines around late March 2027 (inferred from the pattern; not announced).
- Review process. "1+1/2 peer-reviewing system": papers that convince all reviewers are accepted, possibly with minor revisions; papers with overall negative reviews are rejected; borderline papers get a "second rapid review process" after a revised paper and rebuttal, decided by the paper chairs (flowchart: Full paper submission, To Reviewers, accept / borderline / reject, Revised paper and rebuttal due, To Paper Chairs, accept / reject). The DAFx20in22 authors page adds that the second review checks that the revision "provides convincing sound examples if it applies". The DAFx24 best-paper certificates state the award "was judged by the DAFx24 Publication Chairs according to the highest commendation received in the peer review process", so review scores drive awards directly.
- After acceptance. Extended versions of the best papers are invited to a JAES special issue (DAFx24 CFP; DAFx25 special issue "New Frontiers in Digital Audio Effects", guest editors Cecchi, Rau, Bernardini, Hawley, deadline 15 December 2025, https://dafx25.dii.univpm.it/jaes-special-issue/).

## 4. Reviewer guidelines and common weaknesses

No public DAFx reviewer guidelines, review form, scoring rubric, acceptance-rate statement, or program-chair report could be found for DAFx23 through DAFx26 (searched dafx.de, the four conference sites, EasyChair CFP pages, and the auditory list postings). The only reviewer-facing material in public is the 1+1/2 process description and flowchart above, the statement that borderline revisions should provide "convincing sound examples", and the award certificates' wording. Treat the next list as inferred from what the accepted papers do and from the explicit limitation statements in them; it is my reading, and no DAFx document says these are what reviewers flag:

1. No audio examples or companion page (every accepted learned-effect paper has one; the review process names sound examples explicitly).
2. Baselines missing or not named; no ablation of the proposed component (accepted papers label ablations in the results table).
3. Objective metrics only, with no argument for why they track perception, and no listening test or an honest deferral.
4. Listening-test reporting without standard, participants, statistics and anchors (Dal Santo 23 shows the expected level of detail).
5. Training and implementation details too thin to reproduce (optimizer, rates, segment lengths, sample rate, loss weights are always stated).
6. Unstated computational cost or real-time feasibility (operation counts, timing tables and plugin builds are common evidence).
7. Over-claiming beyond the test set, or a single training run presented as conclusive (WildFX pre-empts this explicitly).
8. Limitations not stated or stated generically.
9. Related work that ignores the DAFx lineage on the same effect.
10. Format violations: over 8 pages, non-template fonts or margins, identifying information in a blind submission, captions in the wrong place.

## 5. What this means for the remaster paper (inferred)

- Open the introduction with one paragraph on why finished stereo mixes need restoration and re-mastering, cite the restoration lineage (Godsill and Rayner; Moliner's BABE and BABE-2; declipping surveys) and the DAFx dynamics lineage (Wright and Välimäki 2022 grey-box DRC; Yu et al. 2024 torchcomp; Thompson and Heilemann 2026), state the gap, list contributions, and end with a roadmap paragraph.
- Keep related work inside the introduction or to half a page.
- Put a block diagram of the hybrid chain as Fig. 1, with DSP blocks and learned blocks visually distinguished, and number every equation with a "where" sentence.
- Report evaluation on full mixes with metrics the community already uses on mixes (loudness and LUFS difference, crest factor, FAD with named embeddings, multi-resolution STFT, PEAQ or HAAQI with the He et al. caveats), plus a MUSHRA or MOS test reported at the Dal Santo level of detail, and a companion page with audio.
- State cost (real-time factor, latency, hardware) in a table.
- Write limitations as specific sentences in Discussion or Conclusions.
- Use the DAFx26 template now (`blind` option), 9 pt, 8 pages including references, booktabs tables with captions above, figure captions below; switch to `print` for camera-ready; put the funding line in Acknowledgments; do not post an arXiv preprint before the decision unless labelled as non-peer-reviewed.

## 6. Sources

- DAFx paper archive and API: https://dafx.de/paper-archive/
- DAFx26 authors page: https://dafx26.mit.edu/authors/ ; CFP: https://dafx26.mit.edu/call-for-papers ; template zip: https://dafx26.mit.edu/assets/DAFx26_LaTeX_v3.zip ; flowchart: https://dafx26.mit.edu/assets/decision.png
- DAFx25 site, CFP, program, proceedings, JAES issue: https://dafx25.dii.univpm.it/ , https://dafx25.dii.univpm.it/call-for-papers/ , https://dafx25.dii.univpm.it/detailed-program/ , https://dafx25.dii.univpm.it/proceedings/ , https://dafx25.dii.univpm.it/jaes-special-issue/
- DAFx24 site, CFP, proceedings listing: https://dafx24.surrey.ac.uk/ , https://dafx24.surrey.ac.uk/call-for-papers/ , https://dafx24.surrey.ac.uk/proceedings/ , https://dafx24.surrey.ac.uk/wp-content/uploads/2024/09/Proceedings_of_DAFx24_ForWeb.html
- DAFx23 site, awards, proceedings PDF: https://dafx23.create.aau.dk/ , https://dafx23.create.aau.dk/index.php/awards/ , https://dafx23.create.aau.dk/wp-content/uploads/2023/09/DAFX23_Proceedings.pdf
- DAFx27 site: https://dafx27.deib.polimi.it/
- Award reports: https://www.acoustics.ed.ac.uk/?p=2054 , https://www.aalto.fi/en/news/double-triumph-aalto-acoustics-researchers-win-big-in-italy , https://www.lms.tf.fau.eu/best-paper-award-at-dafx26/ , https://audiolabs-erlangen.com/news/articles/ThreeBestPaperAwards , https://www.acoustics.ed.ac.uk/2024/09/best-presentation-award-for-alistair-carson-at-dafx-24/
- DAFx20in22 authors page (review-process wording about sound examples): https://dafx2020.mdw.ac.at/authors
