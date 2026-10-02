# Mastering practice: a quantitative reference for Lacquer

Compiled 2026-10-02. Purpose: ground Lacquer's decision rules in what mastering engineers and delivery specs say, with numbers, and with a source on every claim.

## 0. How to read this document

Every factual claim carries a citation key in square brackets. Keys resolve to the BibTeX block in section 10. Each key also carries a verification tag:

| Tag | Meaning |
|---|---|
| **[P]** | Primary text read in this session (PDF or page text fetched and the passage located). |
| **[A]** | Bibliographic record and abstract verified (Crossref, OpenAlex, publisher or repository page). Body not read, so only abstract-level claims are used. |
| **[S: x]** | Number taken from a secondary source `x` that quotes or summarizes the primary. The secondary text was read in this session. |
| **[UNVERIFIED]** | Could not confirm against a primary or official source. Do not cite in the paper without checking. |

Two cautions that apply throughout:

1. Practitioner sources (Katz, Ludwig, iZotope, Sound on Sound) give starting points and habits. They are opinions from experienced engineers. Measured studies (Deruty, Pestana, Elowsson, Gerdes, De Man, Kirchberger) give statistics on released music. The two kinds of evidence are labeled separately.
2. Streaming platforms change their normalization behavior without notice. Only Spotify, Apple (Digital Masters and the Atmos asset guide) and Netflix numbers below come from the platform's own documentation. Tidal, YouTube, Amazon and Deezer numbers come from third-party reports and are flagged.

Section 8 contains my own synthesis for Lacquer. It is marked as derived and should be read as a proposal.

---

## 1. The mastering signal chain

### 1.1 Order of processing and the reasons given for it

There is no canonical order. The iZotope mastering guide (2015 edition, revised by Jonathan Wyner) says "there really isn't any single 'correct' order" and gives the author's preferred order as: (1) equalizer, (2) dynamics, (3) post equalizer, (4) harmonic exciter, (5) stereo imaging, (6) loudness maximizer, noting that the exciter and imager are used "less frequently". The one near-rule it states: the maximizer (limiter) and dither go last [izotope2015guide, P].

Bob Katz's rules on the tail of the chain are stricter. Process at the highest sample rate and wordlength available, do sample rate conversion as the penultimate step, and dither to 16 bit as the final step: "Dithering to 16 bit should always be the last step" [katz_faq_dither, P]. His stated reason is that EQ, compression and limiting sound better at the higher rate and wordlength because quantization distortion is spread over a wider bandwidth, and the part above 20 kHz is discarded on conversion [katz_faq_src, P].

Eric James (Sound on Sound, March 2015) describes a typical commercial workflow in two gain stages: run the processing chain and capture "a louder but, as yet, unlimited version of the master", then add level and limiting afterwards. The unlimited capture doubles as the vinyl master. For loud digital masters he adds some light limiting in the first pass because "You can't simply take an unlimited file and add 4 or 6 dB of limiting without sonic consequences" [james2015vinyl, P].

A consolidated order that is consistent with these three sources:

1. Inspection and repair (DC, clicks, noise, clipped input). Only if audible.
2. Corrective EQ (subtractive first), including low-end cleanup.
3. Dynamics (broadband compression, sometimes multiband or parallel). Optional.
4. Tonal EQ after dynamics ("post EQ"). Optional.
5. Saturation or exciter. Rare.
6. Stereo image adjustment. Rare.
7. Gain staging, optional clipper, then true-peak limiter.
8. Sample rate conversion to the delivery rate.
9. Dither (and optional noise shaping) to the delivery wordlength. Once.
10. Sequencing, spacing, fades, album-level loudness balance, metadata.

Steps 1-7 are a synthesis. Steps 8-9 and "limiter and dither last" are stated directly by the sources above.

### 1.2 Stage-by-stage parameters

| Stage | What engineers do | Numbers stated by the source | Source |
|---|---|---|---|
| Corrective EQ | Small broadband moves. Cut before boosting. Few bands. Wide Q for boosts, narrow for cuts. | Typical moves of +/-0.5 to 1.5 dB. "Cutting or boosting more than 2-4 dB means you probably have a problem that you can't fix from the stereo master." | [izotope2015guide, P] |
| Corrective EQ | Same idea, stated more tightly. | "EQ adjustments tend to be about 1 dB or less on a specific frequency range." | [galindo2022eq, P] |
| Corrective EQ | Example of a minimal fix. | "a cut of 0.5dB at 7.4kHz made all the difference" | [wright2025mastering, P] |
| EQ type | Minimum phase moves instruments forward or back in the soundstage and sounds more aggressive. Linear phase keeps depth but disperses transients (pre-echo), worse for bells than shelves and worse at high Q. | Qualitative tradeoff only. | [katz_faq_linphase, P] |
| Mid/side EQ | Filtering the S channel narrows the image in that band and moves content to the center. | "If you filter out some frequencies in the S channel, the sound gets narrower in that frequency range and moves towards the middle." | [katz_faq_ms, P] |
| Low-end cleanup: rumble | High-pass or low shelf, used with caution. | "Shelf or high-pass filters below 30 Hz can get rid of low-frequency rumble and noise, but it comes with a price." | [izotope2015guide, P] |
| Low-end cleanup: band limiting | No routine band limiting. | "There is no rule. We only filter the frequencies that are necessary." On a 25 Hz / 18 kHz roll-off habit: "Never low cut without being sure it is a help." | [katz_faq_removing, P], [katz_faq_bass, P] |
| DC offset | Usually left alone. Treated only if it causes an audible click at start or stop. Then a linear-phase high-pass. | Katz's preferred filter: 13.8 Hz linear-phase HPF. A 20 Hz filter "sometimes eats into important low frequency information". "Very rarely is DC offset a problem in the first place. So I think the most successful and transparent method is not to do anything!" | [katz_faq_dc, P] |
| Low-frequency mono | See section 2.9. Narrow low bands, widen high bands if anything. | No cutoff frequency stated by the iZotope guide. "You may even want to try narrowing of lower bands to pull bass to the center." | [izotope2015guide, P] |
| De-essing, dynamic EQ | Sibilance treated with a detector focused on the sibilant band. | Sibilant bursts "are usually focused somewhere in a region from 4-10kHz". Detector high-pass "at around 4kHz" as a first try. (Written about vocal tracks at mix time.) | [senior2009deessing, P] |
| Broadband compression | Low ratio, high threshold, small gain reduction. Often skipped. | Ratio 1.1:1 to 2.0:1 for a full mix. Threshold set for 1-3 dB of gain reduction as a starting point, "no more than 2-3 dB of compression at the maximum". Attack start point 20-30 ms. Release start point 250 ms. Club tracks "might benefit from higher ratios". | [izotope2015guide, P] |
| Broadband compression | Not always used. | "compression doesn't always play a role in every mastering session, because not every song needs it." A longer attack lets kick transients through. | [galindo2022compression, P] |
| Broadband compression | Hardware and material dependent. | "1 dB on an entire mix with the wrong unit can be severely degrading". "only a couple or 3 dB of compression will cause audible pumping or breathing effects" in many instances. Pumping depends on attack and release relative to tempo and spectrum. | [katz_faq_gr, P] |
| Multiband compression | Used to stop bass from modulating the rest of the mix. Divisive. | Katz: multiband "can reduce pumping by, for example, keeping the bass fraction from modulating the vocal, but I'm not a fan of multiband compression". | [katz_faq_gr, P] |
| Parallel compression | Mix dry with a compressed copy. Raises quiet passages, keeps loud dynamics. | AES definition: "parallel compression amplifies quiet sounds while preserving most of the dynamics of loud sounds", "particularly useful for classical music". If the compressor adds delay, the dry path must be delayed to match. | [aes_td1008, P] |
| Saturation, exciter | Occasional. Easy to overdo. | "It's very easy to overdo an exciter." Listed among processors used "less frequently". | [izotope2015guide, P] |
| Stereo width | Occasional. Check mono after any change. | "You can generally do more widening of higher bands, as opposed to lower bands". "most recordings have phase correlations in the 0 to +1 region". | [izotope2015guide, P] |
| Clipper before limiter | Shave isolated transient peaks so the limiter reacts less. Common on drum-heavy material. | No numbers in the source. "by using a clipper to control the highest peaks you can get more consistent results from a limiter used downstream". Oversampling recommended to avoid aliasing. | [productionexpert2025clippers, P] |
| Clipper before limiter | Typical amount. | 1-3 dB shaved [UNVERIFIED: only found on non-authoritative blogs]. | n/a |
| Limiter: amount | Small amounts are transparent. Large amounts are audible. | Katz: peak limiting with "very quick release time, very quick attack time, high ratio, 1 to 3 dB or so gain reduction can be very invisible". iZotope: "try a threshold that gives you 1 - 3 dB of limiting". James: adding "4 or 6 dB of limiting" to an unlimited file has "sonic consequences". | [katz_faq_gr, P], [izotope2015guide, P], [james2015vinyl, P] |
| Limiter: release | Longer release for heavier limiting. | "More aggressive loudness maximizing (i.e. lower threshold values) will generally require longer release times." | [izotope2015guide, P] |
| Limiter: lookahead and oversampling | Short lookahead keeps transients and loudness but distorts. Long lookahead is safer and duller. | Lookahead below 0.1 ms approximates hard clipping. "choosing 4x oversampling in combination with a minimum lookahead time of 0.1 ms, keeps inter-sample peaks within a range of 0.1 dB." True-peak limiting adds about 5 ms latency in that product. | [fabfilter_prol2, P] |
| Limiter: ceiling | See section 2.3. | -0.3 to -0.8 dB for lossless delivery, -1 to -1.5 dB when the file will be encoded to AAC or MP3 (2015 advice). Never above -0.3 dB. | [izotope2015guide, P] |
| Sample rate conversion | One conversion, near the end, with a high-quality converter. | Katz: "stay at 48 kHz sampling and 24 bit until all of your work is done"; convert "at only the last stage". Apple asks for the highest native sample rate and converts with its own SRC. | [katz_faq_src, P], [apple2021adm, P] |
| Dither | Applied once, last, when reducing wordlength. TPDF. | Lipshitz et al. recommend "nonsubtractive, triangular-pdf dither of 2-LSB peak-to-peak amplitude", which removes distortion and noise modulation at the cost of a noise floor 4.8 dB higher than the undithered quantization noise. | [lipshitz1992dither, P] |
| Dither | Not needed for 24-bit delivery to Apple. | Apple encodes from 24-bit sources through a 32-bit float intermediate, "eliminating the need for adding dither". | [apple2021adm, P] |
| Noise shaping | Moves dither noise to less audible bands. | Standard references are Lipshitz, Vanderkooy and Wannamaker (JAES 1991) and Wannamaker (JAES 1992) [UNVERIFIED: not confirmed in Crossref or OpenAlex this session; check the AES E-Library before citing]. | n/a |
| Fades, sequencing | Done in mastering at long wordlength. Album heard in order before final levels are set. | Katz asks mixers to leave "the segues, fadeouts and gain changes for the mastering house". Problems "sometimes don't become apparent until the album is assembled in its intended order". | [katz_dither_article, P], [katz_audio_mastering, P] |
| Album loudness | Tracks sit at deliberate relative levels. | "the loudest tracks are typically 2 LU louder than the average loudness of their albums". | [aes_td1008, P] |

### 1.3 Monitoring level

Katz's K-System ties metering to a calibrated monitor gain: pink noise at -20 dBFS RMS on one loudspeaker reads 83 dB SPL (C-weighted, slow). He reports that typical pop mastering in 1996 ran the monitor about 6 dB below that reference [katz_levelpractices2, P] [katz2000ksystem, A]. The iZotope guide recommends monitoring "at around 85dB SPL (C-weighted)" [izotope2015guide, P]. This matters for an automatic system only indirectly: engineers judge tonal balance at a fixed loudness, so any tonal analysis should be loudness-normalized first.

---

## 2. Measurements and target values

### 2.1 How loudness, true peak and loudness range are defined

- **Integrated loudness** (ITU-R BS.1770-5, November 2023): K-weighting (a shelving stage plus a high-pass stage), mean square per channel, channel-weighted sum, then gating over 400 ms blocks with 75% overlap. Absolute gate at -70 LKFS, relative gate 10 dB below the absolute-gated level. LKFS and LUFS are the same unit [itu_bs1770_5, P] [ebu_r128, P].
- **True peak** (same recommendation): estimated by 4x oversampling (48 kHz to 192 kHz) and reading the absolute peak, in dBTP [itu_bs1770_5, P]. AES TD1008 notes that true-peak meters "typically have an error of less than 0.6 dB" [aes_td1008, P]. EBU R 128 allows a measurement tolerance of +/-0.3 dB [ebu_r128, P].
- **Loudness range, LRA** (EBU Tech 3342): distribution of short-term loudness (3 s window), absolute gate -70 LUFS, relative gate -20 LU, LRA = 95th percentile minus 10th percentile. Unsuitable for programmes shorter than about 1 minute [ebu_tech3342, P] [ebu_r128, P].
- **PLR** (peak to loudness ratio): maximum true peak minus integrated loudness. Describes level variation "on a microscopic scale", while LRA describes it "on a macroscopic scale" [ebu_r128s2, P] [aes_td1008, P].
- **PSR** (peak to short-term loudness ratio): true peak minus short-term loudness, a running 3 s version of PLR [meterplugs2017psr, P] [shepherd2017psr, A].

### 2.2 Integrated loudness targets by destination

| Destination | Target | True peak | Notes | Source and status |
|---|---|---|---|---|
| Spotify | -14 LUFS normalization (Normal). Loud = -11, Quiet = -19. | Recommends <= -1 dBTP. If the master is louder than -14 LUFS, <= -2 dBTP. | Album normalization when an album is played in order, track normalization on shuffle and playlists. Quiet tracks get positive gain but only up to 1 dB below full scale (example given: -20 LUFS with -5 dBFS peak is lifted to -16 LUFS). In Loud mode a limiter engages at -1 dB with 5 ms attack and 100 ms decay. | [spotify_loudness, P] official |
| Apple Music (Sound Check) | -16 LUFS reported | Apple asks for "at least 1 dB of headroom" | Apple's own brief describes Sound Check and per-album mode but gives no LUFS number. The -16 LUFS figure comes from MeterPlugs (March 2022), which says Apple follows AES TD1008. | [apple2021adm, P] for headroom. [meterplugs2022apple, P] for -16. Official -16 LUFS figure: [UNVERIFIED in Apple documentation] |
| Apple Music, Dolby Atmos | Must not exceed -18 LKFS (BS.1770-4) | Must not exceed -1 dBTP | Atmos must be made from multitracks or stems. "Extracting stems ('de-mixing') from a stereo release is not allowed." | [apple_assetguide_atmos, P] official |
| YouTube | -14 LUFS reference | n/a | Does not turn quiet content up. | [meterplugs2019youtube, P] third-party. No official YouTube document found: [UNVERIFIED officially] |
| Tidal | Album normalization, loudest track of the album at -14 LUFS | n/a | Adopted from Grimm's study of 4.2 million albums. In a test with 38 subjects and 24 songs, 80% preferred album normalization. Does not turn quiet content up. | [audioxpress2017tidal, P], [grimm2019albums, A], [meterplugs2019youtube, P] third-party |
| Amazon Music | -14 LUFS | -2 dBTP often quoted | -14 LUFS per MeterPlugs (October 2019). The -2 dBTP figure appears only on non-official blogs. | [meterplugs2019amazon, P] third-party. -2 dBTP: [UNVERIFIED] |
| Deezer | -15 LUFS | n/a | Per MeterPlugs (October 2019). | [meterplugs2019amazon, P] third-party |
| AES recommendation for streaming services | Music, track-normalized: -16 LUFS. Music, album-normalized: loudest track at -14 LUFS. Speech: -18 LUFS. | <= -1 dBTP at the codec input for lossy streams | Album normalization "strongly recommended", even for shuffle. Keep content above -20 LUFS. Classical: album-normalized, example loudest track at -18 LUFS, no upward normalization. Addressed to distributors. It gives no instruction to mastering engineers. | [aes_td1008, P] |
| EBU R 128 (European broadcast) | -23.0 LUFS | <= -1 dBTP in production (linear PCM) | +/-1.0 LU tolerance where the target is not achievable in practice (live). +/-0.2 LU measurement tolerance in QC. Lower TP limits for data-reduced distribution. | [ebu_r128, P] |
| EBU R 128 s2 (broadcaster streaming) | Stream at -23 LUFS, or an interim -20 to -16 LUFS when no metadata is used | Device-side true-peak limiting if the target is raised | Recommends album or anchor metadata for music services. | [ebu_r128s2, P] |
| ATSC A/85:2013 (US broadcast) | -24 LKFS, anchor element (usually dialogue) | Below -2 dB TP | Variations of about +/-2 dB are acceptable measurement uncertainty and should not be targeted. | [atsc_a85, P] |
| Netflix (near field) | -27 LKFS +/-2 LU, dialog-gated, BS.1770-1, full programme | Must not exceed -2 dBTP | See section 7. | [netflix_soundmix, P] official |

What engineers say about targeting these numbers:

- Spotify's own page says to "Target the loudness level of your master at -14dB integrated LUFS" [spotify_loudness, P]. Working engineers push back on this. Ian Shepherd (via MeterPlugs): "no-one would normally master a heavy rock tune at the same loudness as an acoustic ballad, so why would we start now? The streaming services do the normalization so we don't have to" [meterplugs2019youtube, P].
- Apple: "you should always mix and master your tracks in a way that captures your intended sound, regardless of playback volume" [apple2021adm, P].
- Measured behavior of 218,109 uploads to an analysis platform (amateur and semi-pro producers): masters cluster around -14 LUFS, about 79% of masters are louder than -14 LUFS and 91.55% are louder than -16 LUFS. Mixes peak around -23 LUFS [mourgela2024trends, P].
- Bob Ludwig (2016): "The loudness war. There's no need for it. Streaming services use 'loudness normalization.' ... The louder you make it, the more it will turn the sound down" [ludwig2016mc, P].

### 2.3 True-peak ceilings and why -1 or -2 dBTP

- A lossy codec's decoded output can peak higher than its input. AES TD1008: "High-rate (e.g., 256 kbps) coders may work satisfactorily with as little as -0.5 dBTP for the limiting threshold. Typically, peak overshoot increases as the bit rate decreases, so the limiting threshold may need to be reduced below the recommended -1.0 dBTP" [aes_td1008, P].
- EBU Tech 3343: maximum true peak in production -1 dBTP for linear audio. "For data reduction systems (like MPEG 1L2 and Dolby AC-3) the limit is -2 dBTP" [ebu_tech3343, P].
- ATSC A/85: below -2 dB TP "to provide headroom to avoid potential clipping due to downstream processing (such as audio coding used in delivery)" [atsc_a85, P].
- Spotify: -1 dBTP "is best for lossy formats (Ogg/Vorbis and AAC)". For masters louder than -14 LUFS, -2 dBTP, because "Louder tracks are more susceptible to extra distortion when encoded" [spotify_loudness, P].
- Apple: oversampling in playback DACs can clip a signal at 0 dBFS. "Our recommendation is to leave at least 1 dB of headroom" [apple2021adm, P].
- Other causes of overshoot named by the AES document: filters and sample rate converters downstream of the limiter [aes_td1008, P].

Decision-relevant reading: -1.0 dBTP is the consensus ceiling for a normal-loudness master. -2.0 dBTP is the conservative ceiling for a hot master (louder than about -14 LUFS) or a low-bitrate codec path.

### 2.4 Loudness range (LRA)

- EBU deliberately sets no LRA limit for broadcast: "it is impossible and not useful to define a strict limit for LRA" [ebu_tech3343, P]. Programmes above roughly 20 LU are treated as "very wide" [ebu_tech3343, P].
- Netflix best practice (a recommendation, outside the hard spec): programme LRA between 4 and 18 LU for both 5.1 and 2.0, dialogue LRA of 10 LU or less [netflix_soundmix, P].
- Popular music 1967-2014 (7200 tracks): LRA did not fall during the loudness war and is "relatively independent from both genre and year of release". The weighted mean variance of LRA was about the same grouped by year (14.5 LU) or by genre (14.3 LU), unlike every other dynamics descriptor [deruty2015mir, P]. The earlier 4500-track analysis reached the same conclusion: "the loudness war did not cause any reduction in level variability" at time scales of 3 s and longer [deruty2011sos, P] [deruty2014dynamic, A].
- A lower LRA after processing than before signals that dynamics reduction happened in the chain. EBU recommends LRA for exactly this check [ebu_tech3343, P].
- Per-genre LRA medians (for example classical around 14 LU, EDM 3-6 LU) appear on vendor blogs. I found no peer-reviewed table. Treat such numbers as [UNVERIFIED].

### 2.5 PLR, PSR, crest factor, DR

Measured trends:

- RMS level of best-selling pop rose about 5 dB from the 1970s to the 2000s, with steady growth between 1982 and 2005. A simplified crest factor fell about 3 dB from the early 1980s. Corpus: 4500 tracks, 1969-2010 [deruty2011sos, P].
- In the 7200-track corpus the loudness war peaked in 2007 (RMS and BS.1770 loudness), 2008 (crest factor) and 2006 (density of near-full-scale samples). Significant limiting (more than 3 dB) "seems to have been applied on 33% of all tracks from our corpus, and on 65% of tracks released after 1994" [deruty2015mir, P]. The 2014 JAES paper dates the peak to 2004 on its corpus [deruty2014dynamic, A].
- Dynamics descriptors depend more on release year than on genre. Weighted mean variance within a year versus within a genre: RMS 9.03 vs 14.2 dB, loudness 4.57 vs 7.41 LU, crest factor 1.35 vs 2.25 dB. The amount of limiting "can be considered as independent from genre" [deruty2015mir, P].
- Katz, quoted by Vickers: LP levels rose perhaps 4 dB in 40 years, average CD levels rose almost 20 dB in 20 years [vickers2010loudness, P].

Practitioner guidelines:

- Crest factor. Robert Dennis, quoted by Vickers: a crest factor of 10 dB is "a level of reasonability in pop sound quality", and loudness-war masters reduce it "to as low as 6 dB" [vickers2010loudness, P]. Katz: K-14 practice yields "approximately a 14dB peak to average ratio", and "pop music with a crest factor much less than 14 dB should not be mastered to peak to full scale, as it will sound too loud" [katz_compression_article, P] [katz_levelpractices2, P]. Katz to a dance producer whose unmastered mixes measured 13-17 dB crest factor (flat RMS): "measurably is excellent" [katz_faq_dance, P].
- PSR. Dynameter manual (Ian Shepherd): minimum PSR of 8 is the floor "when mastering, in any genre". Presets: Limited 8, Competitive 10, Balanced 12, Wide 14. A loudness-war casualty "might measure Minimum PSR 6, PLR 8 or even lower". A healthy loud rock or pop song "might have Minimum PSR 9, PLR 11". "if the typical PSR reading of your music is less than 10, it will be turned down by all the online streaming platforms" [shepherd_dynameter, P].
- TT DR value. The community DR database FAQ: "For rock/metal a DR of 8 and above is considered okay. Electronic music can still sound okay with DR 5 because it is less dense" [drdb_faq, P]. Vickers measured DR 3 for a Death Magnetic track and DR 15 for a 1909 cylinder recording [vickers2010loudness, P].
- AES TD1008: "A recording with high peak to loudness ratio (PLR) is often perceived as clearer and less fatiguing than one that has been excessively peak-limited" [aes_td1008, P].

Listening-test evidence is mixed, which matters for how hard a rule Lacquer should encode:

- Croghan et al. (rock and classical samples): a small amount of compression was preferred when loudness was not equalized, "but the highest levels of compression were generally detrimental to quality, whether loudness was equalized or varied" [croghan2012quality, A].
- Hjortkjaer and Walther-Hansen: listeners showed no preference between original and more compressed remasters; listeners are "less sensitive than commonly believed to even high levels of compression" [hjortkjaer2014perceptual, A].
- Ronan et al.: 20 mastering engineers discriminated hyper-compressed versions in 17 of 24 ABX conditions on average. Audibility depended on the crest factor of the music more than on the amount of crest factor reduction, which suggests a threshold of audibility [ronan2017hypercompression, A].
- Wilson and Fazenda: quality ratings of commercial recordings were most associated with signal features related to loudness and dynamic range compression, while liking tracked familiarity [wilson2016perception, A].

### 2.6 The K-System

Three meter scales that put 0 dB at -20, -14 or -12 dBFS (RMS-style average), each tied to the same calibrated monitor level of 83 dB SPL per channel (C-weighted, slow, pink noise at -20 dBFS RMS) [katz_levelpractices2, P] [katz2000ksystem, A].

| Scale | 0 dB point | Intended material (Katz's words) |
|---|---|---|
| K-20 | -20 dBFS | "wide dynamic range material, e.g., large theatre mixes ... audiophile music, classical (symphonic) music" |
| K-14 | -14 dBFS | "the vast majority of moderately-compressed high-fidelity productions intended for home listening (e.g. some home theatre, pop, folk, and rock music)" |
| K-12 | -12 dBFS | "productions to be dedicated for broadcast" |

Mastering practice on the K meter: keep the average from exceeding 0 with occasional peaks to +3 [katz_compression_article, P]. Katz now considers the metering half of the K-System "superceded by a good LUFS meter" and suggests setting 0 LU to -23, -20, -16, -14 or -12 LUFS depending on the application [katz_faq_truepeak, P].

### 2.7 Tonal balance targets

Measured long-term average spectra (LTAS):

- Pestana et al. analyzed commercial recordings 1950-2010 and found "a consistent leaning towards a target equalization curve that stems from practices in the music industry, but also to some extent mimics natural, acoustic spectra of ensembles" [pestana2013spectral, A]. Elowsson and Friberg report that study as 772 recordings with a slope of "approximately 5 dB/octave on average", stated for the range 100 Hz to 4 kHz [pestana2013spectral, S: elowsson2017ltas]. The often-repeated additions (steeper above 4 kHz, low cut near 60 Hz) come from secondary summaries of the same paper and were not checked against the full text: [UNVERIFIED].
- Elowsson and Friberg, 12,345 popular-music tracks, power spectral density smoothed to 1/6 octave, each track loudness-normalized [elowsson2017ltas, P]:
  - Mean LTAS rises up to about 100 Hz, then falls with a slope that steepens with frequency.
  - Slope about 5 dB/octave at 800 Hz and 7.6 dB/octave at 3.2 kHz.
  - A single straight-line fit from 94 Hz to 15.7 kHz gives 5.79 dB/octave, but a quadratic fit is much better (residual norm lower by a factor of 4.1).
  - Between 89 Hz and 4.5 kHz the slope is 4.53 dB/octave regardless of how percussive the track is (standard deviation 0.055 across 11 percussion groups). This is the origin of the "roughly -4.5 dB/octave" figure.
  - Variation across tracks is smallest at 200 Hz - 1 kHz and fairly small at 1-4 kHz. It is largest at the low and high extremes.
  - Tracks with more percussion have relatively more energy in the bass and in the highs. The authors argue that genre differences in LTAS are "mainly a side-effect of percussive prominence".
- These slopes are in power spectral density per Hz. Pink noise falls 3 dB/octave on that scale, so the pop average is about 1.5 dB/octave steeper than pink over 89 Hz - 4.5 kHz, and it would tilt down by about that much on a constant-percentage-bandwidth (RTA) display where pink noise reads flat. (My arithmetic. The paper does not state this.)
- Bass has grown over time. Billboard Hot 100 songs 1955-2016: RMS energy and loudness increased, and "when controlling for overall RMS, only the lowest frequency bands showed an increase over time" (0-100 Hz) [hove2019bass, P].

Practitioner view:

- Ludwig: averaging consumer speakers gives a nearly flat curve, so mastering on an accurate flat system "yields a product that sounds correct on more systems" [ludwig2011castle, P].
- Katz: make the mix "as neutral as possible"; small bass-boosting consumer systems act around 70-100 Hz and need no compensation [katz_faq_eqfinal, P].
- Elowsson and Friberg cite Katz for the idea that mastering engineers keep the "symphonic tonal balance" as a basic reference for most pop, rock, jazz and folk [S: elowsson2017ltas, citing the 2002 first edition of Katz's book; book text not read].

### 2.8 Stereo correlation and mono compatibility

- Correlation meter scale: +1 for identical channels (mono), 0 for fully different channels (widest), -1 for polarity-inverted channels. "any steady reading in the negative half indicates a reduced degree of mono-compatibility: something will get lost (or attenuated) when you listen in mono" [robjohns2016correlation, P].
- "In general, most recordings have phase correlations in the 0 to +1 region." A brief dip below zero is not necessarily a problem [izotope2015guide, P].
- Check in mono. If "important instruments vanish, or if the level drops significantly, you might need to rethink what you are doing" [izotope2015guide, P].
- Netflix requires the 2.0 Lo/Ro or Lt/Rt mix "to be mono compatible" [netflix_soundmix, P].
- Prevalence in the wild: mono compatibility issues in about 16.9% of mixes and 12.0% of masters; phase issues in 16.3% of mixes and 15.6% of masters. Among mixes, 39.04% were classed narrow and 17.94% wide. Among masters, 39.36% wide and 16.45% narrow [mourgela2024trends, P]. These classes use the platform's own thresholds, which the paper describes only in outline.

### 2.9 Low-frequency mono

- In a study of eight songs, each mixed by eight engineers, the bass was the only source significantly more central than the full mix in the lowest frequency band. The authors explain this "by noting that most of the low frequency sources are panned centre" [deman2014analysis, P].
- Ludwig: "Almost all pop mixes are mixed with the bass and kick drum panned to the center which is proper". For reissues of early stereo recordings with bass locked to one side, "sometimes it is decided to filter the low bass into the center by mono-ing the signal somewhat", a decision he links to earbud listening [ludwig2011castle, P].
- Vinyl origin, with a caveat. Eric James calls the rule that vinyl masters must be mono in the lows (he quotes a label's "80-200 [Hz], almost mono") a myth, pointing to orchestral records with basses hard right. His position: "wildly out-of-phase and excessive bass can be problematic", some taming may be needed, and the decision is best left to the cutting engineer [james2015vinyl, P].
- Streaming relevance: mono summing on small speakers and correlation at low frequencies. The iZotope guide suggests narrowing low bands "to pull bass to the center" [izotope2015guide, P].
- Specific crossover frequencies for bass mono (90, 100, 120, 150 Hz are all quoted on blogs) are [UNVERIFIED]. No authoritative source read in this session states one.

### 2.10 Inter-sample peaks

- Reconstruction between samples can exceed the largest sample value. BS.1770 true-peak metering exists for this reason [itu_bs1770_5, P] [fabfilter_prol2, P].
- Nielsen and Lund showed that inter-sample peaks can sit considerably above 0 dBFS and tested how domestic CD players reproduce such signals [nielsen2000zerodbfs, A]. Lund's "Stop Counting Samples" (AES 2006) is the usual citation for moving from sample-peak to true-peak metering [lund2006stop, bibliographic record verified; content not read].
- Katz: a true-peak meter can read above 0 dBFS; that reading reflects "what will happen in the analog domain or after requantization due to sample rate conversion or other processes" [katz_faq_truepeak, P].
- In the wild: fewer than half of uploaded masters (42.53%) were free of clipping, against 68.58% of mixes. The electronic genre had the highest share of major clipping [mourgela2024trends, P].

---

## 3. Genre differences

What measured studies support:

| Dimension | Finding | Source |
|---|---|---|
| Short-term dynamic range | 1000 songs (100 per genre, released 2000-2014), percentile analysis per IEC 60118-15 (125 ms windows, third-octave bands, 99th minus 30th percentile). Pop, rap, rock and schlager had the smallest dynamic range, then jazz, then classical genres (chamber, choir, orchestra, piano, opera). Speech in quiet exceeded all music genres. | [kirchberger2016dynamic, P] |
| Loudness, crest factor, limiting | In mainstream popular music these depend on release year more than on genre. The amount of limiting is independent of genre. | [deruty2015mir, P] |
| Macro-dynamics (LRA) | About equally variable within a year and within a genre. | [deruty2015mir, P] |
| Low and high frequency balance | Hip-hop, rock, pop and electronic music have louder lows (up to 150 Hz) and louder highs (5 kHz and above) than jazz and folk, which carry relatively more mid-frequency level. Explained largely by percussion level. | [pestana2013spectral, S: elowsson2017ltas], [elowsson2017ltas, P] |
| Vocal level | Lead-vocal-to-accompaniment ratio is positive for country, rap and pop, near zero for rock, negative for metal. Solo artists have higher ratios than bands. | [gerdes2023leadvocal, A] |
| Compression tolerance | For rock and classical samples alike, heavy compression limiting reduced quality ratings. | [croghan2012quality, A] |
| Clipping and tonal faults in uploads | Electronic music shows the most major clipping. Orchestral and metal uploads show excess high-frequency energy more often. | [mourgela2024trends, P] |

What practitioners say, by genre:

| Genre | Loudness and dynamics | Tonal and spatial | Source |
|---|---|---|---|
| Classical | K-20. Album normalization, example loudest track at -18 LUFS, no upward normalization, "large peak-to-loudness ratio". Parallel compression if dynamics must be reduced. | Natural ensemble balance. | [katz_levelpractices2, P], [aes_td1008, P] |
| Jazz, folk, acoustic | K-14 for folk; "audiophile" material on K-20. Wider range of PSR values, fewer low readings. | Relatively more midrange level than pop. | [katz_levelpractices2, P], [shepherd_dynameter, P], [elowsson2017ltas, P] |
| Pop, rock | K-14. Crest factor around 10-14 dB considered healthy. Minimum PSR 8, typical loud song PSR 9 / PLR 11. DR 8+ "okay" for rock. | Bass and kick centered. | [katz_levelpractices2, P], [vickers2010loudness, P], [shepherd_dynameter, P], [drdb_faq, P], [ludwig2011castle, P] |
| Metal | Same minimum PSR as other genres per Shepherd. DR 8+ "okay". Death Magnetic is the standard cautionary example: very low crest factor combined with low loudness range, which Deruty describes as "compact all the time". Vocals sit below the accompaniment (about -3 dB LAR). | Dense guitars tolerate low crest factor worse than sparse percussive music. | [shepherd_dynameter, P], [drdb_faq, P], [deruty2011sos, P], [gerdes2023leadvocal, S: search-engine extract of the article text] |
| Hip-hop, rap | Sparse production with loud kick and snare gives high micro-variability, so low crest factors suit it better than they suit guitar-driven rock. Vocals above the accompaniment. | Louder lows and highs than jazz or folk. | [deruty2011sos, P], [gerdes2023leadvocal, A], [elowsson2017ltas, P] |
| EDM, club | Higher compression ratios may suit club tracks. DR 5 can still sound acceptable "because it is less dense". Same PSR floor per Shepherd. | Louder lows and highs. | [izotope2015guide, P], [drdb_faq, P], [shepherd_dynameter, P] |

Gaps: I found no peer-reviewed per-genre table of integrated loudness, LRA or PLR for commercial masters. A Korean-language study of Billboard year-end number-one singles 1965-2020 reports that LUFS rose about 30% relative to the 1980s-90s and that LRA shrank until the 2000s and widened about 30% after 2010 [choi2021loudness, A]; its abstract gives relative changes only.

---

## 4. Stem mastering and source balance

### 4.1 What engineers do with stems

- Definition: the mix engineer delivers several stereo stems (grouped instruments with their processing printed), and the mastering engineer adjusts the stems before producing the stereo master. Preparation rules: stems must start and end together, and bus processing such as limiting should be removed before export [luthar2023stems, P].
- Typical use is balance correction that a stereo file does not allow: vocal level, low-end balance, EQ of one group without touching the others [luthar2023stems, P].
- Katz makes the same point from the film side: to adapt a wide-range theatrical mix for home use, producers may compress the master "or better, remix the entire program from the multi-track stems (submixes)" [katz_levelpractices2, P].
- Stems must reconstruct the mix. Netflix: "Provide 5.1 Dialog, Music and Effects stems that equal the 5.1 mix when combined" [netflix_soundmix, P].
- Fixing a stereo mix by EQ has limits that stems remove. Katz on an over-loud, boomy bass: isolating the problem frequencies "without causing the bass drum to be lost or suffer or much worse the keyboards, guitars or midrange instruments to suffer can be an exhausting, time-consuming and costly venture" [katz_faq_bassheavy, P].
- A constraint relevant to any system that separates stems from a stereo master: Apple does not accept Dolby Atmos tracks made by de-mixing a stereo release [apple_assetguide_atmos, P]. The rule covers Atmos deliverables. Stereo masters are outside its scope.

### 4.2 Relative loudness of sources in commercial and expert mixes

- De Man et al. (eight songs, each mixed by eight engineers, loudness of each source relative to the whole mix, measured with ITU-R BS.1770 loudness modified for multitrack content: 280 ms time constant and +10 dB pre-filter gain): lead vocal -2.7 +/- 1.6 LU, "suggesting an agreement on a 'target loudness' of about -3 LU relative to the overall mix loudness". The vocal was significantly louder than every other source. From the paper's Figure 1 and Table 2: kick about -13.2 +/- 4.1 LU, snare about -16.8 +/- 6.2 LU, other drums about -12.7 +/- 5.5 LU, bass about -9.5 +/- 3.5 LU. The relative loudness of bass, snare and other drums depended on the engineer; the vocal level did not vary much [deman2014analysis, P]. Note: in the extracted table the Bass and Lead vocal columns appear in the opposite order to the figure. The mapping above follows the figure and the body text.
- Gerdes and Siedenburg (Billboard Hot 100 top songs per year 1946-2020, source-separated): lead-vocal-to-accompaniment level ratio fell from about 5 dB to about 1 dB by around 1975 and has been static since. Positive for country, rap and pop, near zero for rock, negative for metal. Higher for solo artists than for bands [gerdes2023leadvocal, A]. Metal mean about -3.1 dB with interval [-3.6, -2.6] [S: search-engine extract of the article text; confirm against the PDF].
- The two studies agree. A vocal 1 dB above the accompaniment is 2.5 dB below the sum of the two (10 log10 of 1.259 / 2.259 = -2.5 dB), close to De Man's -2.7 LU. (My arithmetic, assuming uncorrelated sources and equal weighting.)
- Pestana and Reiss derived mixing best practices from interviews and mix analysis [pestana2014intelligent, A]. The full text was not available, so no numbers are taken from it here.
- Typical stem balances for drums, bass and "music" stems in released masters, as opposed to expert mixes of eight songs, have no measured source that I could verify. [UNVERIFIED gap]

---

## 5. Common faults and how engineers detect and fix them

| Fault | Where it lives | How it is detected | Typical fix and size | Source |
|---|---|---|---|---|
| Muddiness | 100-300 Hz (iZotope). "Mud" at about 200 Hz, boxiness at 300-500 Hz (Owsinski). | Listening. Spectrum shows excess low-mid energy relative to the reference slope. | Broad cut. Mastering-scale moves are about 0.5-1.5 dB. | [izotope2015guide, P], [owsinski2016trouble, P] |
| Nasal or honky | 250-1000 Hz (iZotope). 1-1.5 kHz "nasal", about 800 Hz cheap-speaker tone (Owsinski). | Listening. | Narrow to moderate cut. | [izotope2015guide, P], [owsinski2016trouble, P] |
| Harshness | 2-3.5 kHz (iZotope). 4-6 kHz presence band in excess makes a track thin or sibilant (Owsinski). | Listening at calibrated level. Fatigue. | "Try cutting this range a few dB" at mix scale. At mastering scale, example of 0.5 dB at 7.4 kHz. | [izotope2015guide, P], [owsinski2016trouble, P], [wright2025mastering, P] |
| Sibilance | 4-10 kHz. | Bursts far above the surrounding level in that band. | Split-band de-esser or dynamic EQ keyed from a detector high-passed near 4 kHz. Heavy de-essing can remove character; Wright declined it on a gritty rock vocal. | [senior2009deessing, P], [wright2025mastering, P] |
| Boominess, low-end build-up | Below about 200 Hz. | Listening on full-range monitors. Common in bass-driven productions. | Low shelf or high-pass, sized to keep fullness. Hard to fix on a stereo file without hurting kick or midrange instruments. | [galindo2022eq, P], [katz_faq_bassheavy, P] |
| Thin low end | Below about 200 Hz. | Listening. "not enough of it can make it sound thin". | Gentle low shelf. Keep deep bass rather than high-passing by habit. | [owsinski2016trouble, P], [katz_faq_bass, P] |
| Rumble, subsonics | Below about 30 Hz. | Spectrum, excess limiter activity. | High-pass or shelf below 30 Hz, only when it helps. | [izotope2015guide, P], [katz_faq_bass, P] |
| DC offset | 0 Hz. | Click or pop when starting or stopping during a soft passage. | Usually nothing. Otherwise a linear-phase high-pass (Katz uses 13.8 Hz). | [katz_faq_dc, P] |
| Phase and correlation problems | Whole band or low band. | Correlation meter sitting below 0. Level drop or vanishing instruments in mono. | Narrow the affected band, reduce S in that band. | [robjohns2016correlation, P], [izotope2015guide, P], [katz_faq_ms, P] |
| Excessive width | Usually mid and high bands after widening. | Correlation trending to 0 and below. Mono check. | Reduce widening. Widening is safer in high bands than low bands. | [izotope2015guide, P] |
| Over-compression | Whole signal. | Low crest factor or PSR (below about 8), pumping, loss of punch. LRA lower after processing than before. | Back off. Katz for remasters: reduce gain so the average falls within K-14, or go back to unprocessed mixes. | [shepherd_dynameter, P], [ebu_tech3343, P], [katz_levelpractices2, P] |
| Clipping | Peaks. | Runs of full-scale samples, true peak above 0 dBTP. Apple supplies `afclip` for checking masters and encodes. | Lower the level into the limiter, set a true-peak ceiling. Clipped sources cannot be fully repaired by gain alone. | [apple2021adm, P], [mourgela2024trends, P] |
| Noise | Broadband. | Listening. Katz does not treat a noise measurement as the criterion. | Often left alone. Removing noise can unmask distortion or remove ambience. Wright declined aggressive noise reduction on an intimate vocal. | [katz_faq_noise, P], [wright2025mastering, P] |
| Clicks from edits or fades | Transients. | Listening at edit points. | Fades and edits done at long wordlength in mastering. | [katz_dither_article, P] |

---

## 6. "Do no harm": documented statements about when not to process

Short quotations, each from a source read in this session.

- Bob Katz on compression: "Rule #1: There are no rules." On bus compression: "compare IN and OUT very carefully, and don't be afraid to patch it OUT if that sounds better" [katz_compression_article, P].
- Bob Katz on DC offset: "the most successful and transparent method is not to do anything!" and "the cure can be worse than the disease" [katz_faq_dc, P].
- Bob Katz on routine filtering: "There is no rule. We only filter the frequencies that are necessary" [katz_faq_removing, P]. "Never low cut without being sure it is a help" [katz_faq_bass, P].
- Bob Katz on adding analog stages: weigh "the cure versus the disease", since every extra conversion costs transparency [katz_audio_mastering, P].
- Bob Katz on loudness: "your hot CD doesn't get any louder for the public; they just turn their monitor down" [katz_compression_article, P]. "hyper-compressed recordings do not play well on the radio" [katz_levelpractices2, P].
- Bob Ludwig on a release that needed little: with Jack White's Blunderbuss "we did little if any mastering limiting. Just the compression that was in the mix ... No one complained that the sound was too low" [ludwig2016mc, P].
- Bob Ludwig on over-compression: at some point the music "has sucked the life out of the music and makes the listener subconsciously not want to hear the music again". He also notes the bias that drives it: listeners pick the version that is 0.5 or 1 dB louder as "sounding best" [ludwig2011castle, P].
- Alexander Wright (Sound on Sound, 2025): "change as little as necessary. If it isn't broken, don't fix it" [wright2025mastering, P].
- iZotope guide: "just because you have all these modules doesn't require that you use them all. Only use as many as you need" [izotope2015guide, P].
- Jett Galindo (iZotope): "not every song needs" compression in mastering [galindo2022compression, P].
- Eric James on the low-end mono rule for vinyl: leave the decision to the cutting engineer [james2015vinyl, P].
- Apple: "We ask that you avoid clipping the signal" and master for the intended sound "regardless of playback volume" [apple2021adm, P].
- EBU: "no parameter and corresponding maximum allowed value can guarantee a good mix!" [ebu_tech3343, P].

Bob Katz's book *Mastering Audio: The Art and the Science* (3rd ed., Focal Press, 2014) is the standard long-form reference for these positions [katz2014mastering, bibliographic record verified]. I did not have the book text, so no page-level quotes from it appear here. The phrase "first, do no harm" is widely attributed to mastering teaching, but I could not tie it to a specific page: [UNVERIFIED attribution].

---

## 7. Film and "cinema-grade" delivery, briefly

- **Theatrical calibration.** SMPTE RP 200:2012: reference level "should be 85 dBC for normal theatrical operation", measured with wideband pink noise at the reference electrical level (20 dB below full modulation), one screen channel at a time. With two surround channels, each is typically set 3 dB below the screen-channel level. The LFE channel shows 10 dB of in-band gain relative to a screen channel [smpte_rp200, P]. Katz traces the 83 dB SPL figure used for music calibration to the same Dolby film practice [katz_levelpractices2, P].
- **Smaller rooms.** ATSC A/85 Table 10.2 scales the reference SPL by room volume: 85 dB above 20,000 cubic feet, 82 dB for 10,000-19,999, 80 dB for 5,000-9,999, 78 dB for 1,500-4,999, 76 dB below 1,500 [atsc_a85, P].
- **Netflix near-field deliverables.** -27 LKFS +/-2 LU dialog-gated (BS.1770-1) over the full programme. Peaks must not exceed -2 dBTP; for Atmos, true-peak limiters on beds and objects at -2.3 dBFS or lower. Mixing reference level 79 or 82 dB SPL. 48 kHz, 24-bit. Stereo mix must be mono compatible. Recommended LRA 4-18 LU, dialogue LRA 10 LU or less [netflix_soundmix, P].
- **Netflix theatrical deliverables.** Mixed at 85 dB SPL reference, "no LKFS loudness requirement and may peak at 0db True Peak", delivered as a separate set from the near-field mix [netflix_soundmix, P].
- **Broadcast.** EBU R 128: -23 LUFS, -1 dBTP. ATSC A/85: -24 LKFS anchored on dialogue, -2 dBTP [ebu_r128, P] [atsc_a85, P].
- **Immersive music.** Apple Music Dolby Atmos: at most -18 LKFS integrated, at most -1 dBTP [apple_assetguide_atmos, P].

The music-relevant point: film and broadcast specs anchor loudness on dialogue and keep 20 dB or more of peak headroom above the anchor. Music streaming targets sit 9 to 13 LU higher (-14 to -16 LUFS against -23 to -27) and leave correspondingly less headroom.

---

## 8. Derived implications for Lacquer (my synthesis)

These follow from the sections above. Each line names the evidence it rests on.

1. **Default ceiling -1.0 dBTP, measured with 4x oversampling. Switch to -2.0 dBTP when the output is louder than -14 LUFS integrated.** Basis: Spotify, AES TD1008, EBU Tech 3343 (2.3).
2. **Do not chase -14 LUFS for every track.** Treat platform targets as the level at which the master will be auditioned. Choose loudness from the material's own crest factor, with a PSR floor of 8 at the loudest section and a preference for 9-10 or more. Basis: Shepherd, Katz, Ludwig, Deruty (2.2, 2.5).
3. **Limiter gain reduction budget: 1-3 dB is the transparent zone. Above about 4 dB, flag it.** Basis: Katz, iZotope, James (1.2).
4. **Compression is optional. If used: ratio 1.1-2.0, 1-3 dB gain reduction, attack 20-30 ms, release about 250 ms as starting points.** Basis: iZotope guide, Galindo, Katz (1.2).
5. **EQ moves on the stereo bus stay within about +/-1.5 dB per band. A wanted change above 2-4 dB signals a mix-level problem, which is where stem-level correction earns its keep.** Basis: iZotope guide, Galindo, Katz bass FAQ (1.2, 4.1).
6. **Tonal target: LTAS slope near -4.5 dB/octave between 89 Hz and 4.5 kHz, steepening above (about -7.6 dB/octave at 3.2 kHz), with a rise up to about 100 Hz. Tolerate wide deviations below 200 Hz and above 4 kHz, and condition the bass and treble expectation on percussion level.** Basis: Elowsson and Friberg, Pestana (2.7).
7. **Vocal stem target: about -2.7 LU relative to the full mix (roughly +1 dB against the accompaniment), shifted by genre: higher for country, rap, pop and solo artists, near 0 dB LAR for rock, about -3 dB LAR for metal.** Basis: De Man, Gerdes (4.2).
8. **No routine high-pass, DC removal, low-band mono, widening, exciter or noise reduction. Each needs a detected fault first.** Basis: section 6.
9. **Correlation guard: keep the long-term correlation at or above 0, check mono-sum level loss, and narrow before widening in low bands.** Basis: Robjohns, iZotope guide (2.8, 2.9).
10. **LRA check: report LRA before and after. A drop is evidence that macro-dynamics were reduced, which a master should avoid unless asked.** Basis: EBU Tech 3343, Deruty (2.4).
11. **Dither once, TPDF at 2 LSB peak-to-peak, only when reducing wordlength, after any sample rate conversion.** Basis: Lipshitz et al., Katz (1.2).

---

## 9. Open items and unverified claims

| Item | Status |
|---|---|
| Apple Music Sound Check target of -16 LUFS | Third-party report only (MeterPlugs 2022). No number in Apple's Digital Masters brief. |
| YouTube -14 LUFS, Amazon -14 LUFS and -2 dBTP, Deezer -15 LUFS | Third-party reports. No official pages found. |
| Pestana 2013 details beyond the abstract (dataset size 772, 5 dB/octave over 100 Hz - 4 kHz, genre contrasts) | Taken from Elowsson and Friberg's description of the paper. The "low cut near 60 Hz" and "steeper above 4 kHz" details were not confirmed. AES full text not accessible. |
| Gerdes and Siedenburg per-genre LAR values | Abstract verified. Only the metal value (-3.1 dB) was seen, via a search extract. Read the PDF (DOI 10.1121/10.0017773) before quoting numbers. |
| Per-genre LRA, PLR and integrated loudness medians for commercial masters | No peer-reviewed table found. |
| Low-frequency mono crossover frequency | No authoritative number found. |
| Clipper amounts (1-3 dB) | Blog-level only. |
| Noise-shaping references (Lipshitz, Vanderkooy, Wannamaker 1991; Wannamaker 1992) | Not confirmed in Crossref or OpenAlex. Omitted from the BibTeX. |
| Katz book page-level quotes; "first, do no harm" attribution | Book not available. Online Katz articles used instead. |
| Shepherd et al. AES e-Brief 373 author order | OpenAlex lists Shepherd, Grimm, Tapper, Kahsnitz, Kerr. Grimm Audio's page lists four authors in a different order. Check the AES E-Library record. |
| AES77 (the Recommended Practice that EBU R 128 s2 cites alongside TD1008) | Existence confirmed only through the EBU reference list. Not read. |
| De Man 2014 Table 2 column order | Figure and body text used. Table header order conflicts. |

---

## 10. BibTeX

All entries below were verified in this session as described in section 0. Web sources carry the access date 2026-10-02. Titles that contain a dash in the original are written with a colon here.

```bibtex
% ---------- Standards and technical documents ----------

@techreport{itu_bs1770_5,
  author      = {{International Telecommunication Union}},
  title       = {Algorithms to measure audio programme loudness and true-peak audio level},
  institution = {ITU Radiocommunication Sector},
  type        = {Recommendation},
  number      = {ITU-R BS.1770-5},
  year        = {2023},
  month       = nov,
  address     = {Geneva}
}

@techreport{ebu_r128,
  author      = {{European Broadcasting Union}},
  title       = {Loudness normalisation and permitted maximum level of audio signals},
  institution = {EBU},
  type        = {EBU Recommendation},
  number      = {R 128, version 5},
  year        = {2023},
  month       = nov,
  address     = {Geneva},
  url         = {https://tech.ebu.ch/docs/r/r128.pdf}
}

@techreport{ebu_r128s2,
  author      = {{European Broadcasting Union}},
  title       = {Loudness in Streaming: Supplement 2 to {R} 128},
  institution = {EBU},
  type        = {EBU Recommendation},
  number      = {R 128 s2, version 3},
  year        = {2023},
  month       = nov,
  address     = {Geneva},
  url         = {https://tech.ebu.ch/docs/r/r128s2.pdf}
}

@techreport{ebu_tech3342,
  author      = {{European Broadcasting Union}},
  title       = {Loudness Range: A measure to supplement {EBU R} 128 loudness normalization},
  institution = {EBU},
  number      = {Tech 3342},
  year        = {2023},
  month       = nov,
  address     = {Geneva},
  url         = {https://tech.ebu.ch/docs/tech/tech3342.pdf}
}

@techreport{ebu_tech3343,
  author      = {{European Broadcasting Union}},
  title       = {Guidelines for Production of Programmes in accordance with {EBU R} 128},
  institution = {EBU},
  number      = {Tech 3343},
  year        = {2023},
  month       = nov,
  address     = {Geneva},
  url         = {https://tech.ebu.ch/docs/tech/tech3343.pdf}
}

@techreport{atsc_a85,
  author      = {{Advanced Television Systems Committee}},
  title       = {{ATSC} Recommended Practice: Techniques for Establishing and Maintaining Audio Loudness for Digital Television},
  institution = {ATSC},
  number      = {A/85:2013},
  year        = {2013},
  month       = mar,
  address     = {Washington, D.C.}
}

@techreport{aes_td1008,
  author      = {{Audio Engineering Society Technical Council}},
  title       = {Recommendations for Loudness of Internet Audio Streaming and On-Demand Distribution},
  institution = {Audio Engineering Society},
  type        = {Technical Document},
  number      = {AESTD1008.1.21-9},
  year        = {2021},
  month       = sep,
  address     = {New York}
}

@techreport{smpte_rp200,
  author      = {{Society of Motion Picture and Television Engineers}},
  title       = {Relative and Absolute Sound Pressure Levels for Motion-Picture Multichannel Sound Systems: Applicable for Analog Photographic Film Audio, Digital Photographic Film Audio and {D-Cinema}},
  institution = {SMPTE},
  type        = {Recommended Practice},
  number      = {RP 200:2012},
  year        = {2012}
}

% ---------- Platform documentation ----------

@misc{spotify_loudness,
  author       = {{Spotify}},
  title        = {Loudness normalization on {Spotify}},
  howpublished = {Spotify for Artists support article},
  url          = {https://support.spotify.com/artists/article/loudness-normalization/},
  note         = {Accessed 2026-10-02}
}

@misc{apple2021adm,
  author       = {{Apple Inc.}},
  title        = {Apple Digital Masters: Technology Brief},
  year         = {2021},
  month        = apr,
  url          = {https://www.apple.com/itunes/docs/apple-digital-masters.pdf},
  note         = {Accessed 2026-10-02}
}

@misc{apple_assetguide_atmos,
  author       = {{Apple Inc.}},
  title        = {Apple Video and Audio Asset Guide: Immersive Audio Source Profile},
  url          = {https://help.apple.com/itc/videoaudioassetguide/en.lproj/itcf946aaace.html},
  note         = {Accessed 2026-10-02}
}

@misc{netflix_soundmix,
  author       = {{Netflix}},
  title        = {Sound Mix Specifications and Best Practices (branded content)},
  howpublished = {Netflix Studio Partner Knowledge Hub},
  url          = {https://studiopartner.netflix.net/studio/branded-sound-mix-spec-and-best-practices},
  note         = {Accessed 2026-10-02. Formerly Netflix Sound Mix Specifications and Best Practices v1.5 on Partner Help.}
}

% ---------- Peer-reviewed and conference papers ----------

@article{katz2000ksystem,
  author  = {Katz, Bob},
  title   = {Integrated Approach to Metering, Monitoring, and Leveling Practices, Part 1: Two-Channel Metering},
  journal = {Journal of the Audio Engineering Society},
  volume  = {48},
  number  = {9},
  pages   = {800--809},
  year    = {2000}
}

@article{lipshitz1992dither,
  author  = {Lipshitz, Stanley P. and Wannamaker, Robert A. and Vanderkooy, John},
  title   = {Quantization and Dither: A Theoretical Survey},
  journal = {Journal of the Audio Engineering Society},
  volume  = {40},
  number  = {5},
  pages   = {355--375},
  year    = {1992}
}

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
  author    = {Deruty, Emmanuel and Pachet, Fran{\c{c}}ois},
  title     = {The {MIR} Perspective on the Evolution of Dynamics in Mainstream Music},
  booktitle = {Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR)},
  address   = {M{\'a}laga, Spain},
  pages     = {722--727},
  year      = {2015}
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
  title   = {Quality and loudness judgments for music subjected to compression limiting},
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
  author  = {Wilson, Alex and Fazenda, Bruno M.},
  title   = {Perception of Audio Quality in Productions of Popular Music},
  journal = {Journal of the Audio Engineering Society},
  volume  = {64},
  number  = {1/2},
  pages   = {23--34},
  year    = {2016},
  doi     = {10.17743/jaes.2015.0090}
}

@article{kirchberger2016dynamic,
  author  = {Kirchberger, Martin and Russo, Frank A.},
  title   = {Dynamic Range Across Music Genres and the Perception of Dynamic Compression in Hearing-Impaired Listeners},
  journal = {Trends in Hearing},
  volume  = {20},
  year    = {2016},
  doi     = {10.1177/2331216516630549}
}

@article{gerdes2023leadvocal,
  author  = {Gerdes, Karsten and Siedenburg, Kai},
  title   = {Lead-vocal level in recordings of popular music 1946--2020},
  journal = {JASA Express Letters},
  volume  = {3},
  number  = {4},
  pages   = {043201},
  year    = {2023},
  doi     = {10.1121/10.0017773}
}

@article{hove2019bass,
  author  = {Hove, Michael J. and Vuust, Peter and Stupacher, Jan},
  title   = {Increased levels of bass in popular music recordings 1955--2016 and their relation to loudness},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {145},
  number  = {4},
  pages   = {2247--2253},
  year    = {2019},
  doi     = {10.1121/1.5097587}
}

@article{choi2021loudness,
  author  = {Choi, Sungrak},
  title   = {An Empirical Study on the Changes in Loudness and Dynamic Range of Pop Music According to Digitalization of Music Content: Focusing on the Change of {LUFS}, {PLR}, and {LRA} of the {Billboard} No.1 Single from 1965 to 2020},
  journal = {The Korean Society of Culture and Convergence},
  volume  = {43},
  number  = {3},
  pages   = {71--96},
  year    = {2021},
  doi     = {10.33645/cnc.2021.03.43.3.71},
  note    = {In Korean}
}

@inproceedings{deman2014analysis,
  author    = {De Man, Brecht and Leonard, Brett and King, Richard and Reiss, Joshua D.},
  title     = {An Analysis and Evaluation of Audio Features for Multitrack Music Mixtures},
  booktitle = {Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR)},
  address   = {Taipei, Taiwan},
  pages     = {137--142},
  year      = {2014}
}

@inproceedings{pestana2013spectral,
  author    = {Pestana, Pedro Duarte and Ma, Zheng and Reiss, Joshua D. and Barbosa, {\'A}lvaro and Black, Dawn A. A.},
  title     = {Spectral Characteristics of Popular Commercial Recordings 1950--2010},
  booktitle = {Audio Engineering Society Convention 135},
  address   = {New York, NY, USA},
  year      = {2013},
  month     = oct,
  note      = {Paper 8960}
}

@inproceedings{pestana2014intelligent,
  author    = {Pestana, Pedro Duarte and Reiss, Joshua D.},
  title     = {Intelligent Audio Production Strategies Informed by Best Practices},
  booktitle = {Audio Engineering Society 53rd International Conference: Semantic Audio},
  address   = {London, UK},
  year      = {2014},
  month     = jan
}

@inproceedings{elowsson2017ltas,
  author    = {Elowsson, Anders and Friberg, Anders},
  title     = {Long-term Average Spectrum in Popular Music and its Relation to the Level of the Percussion},
  booktitle = {Audio Engineering Society Convention 142},
  address   = {Berlin, Germany},
  year      = {2017},
  month     = may,
  note      = {Paper 9762}
}

@inproceedings{vickers2010loudness,
  author    = {Vickers, Earl},
  title     = {The Loudness War: Background, Speculation, and Recommendations},
  booktitle = {Audio Engineering Society Convention 129},
  address   = {San Francisco, CA, USA},
  year      = {2010},
  month     = nov,
  note      = {Paper 8175}
}

@inproceedings{grimm2019albums,
  author    = {Grimm, Eelco},
  title     = {Analyzing Loudness Aspects of 4.2 Million Music Albums in Search of an Optimal Loudness Target for Music Streaming},
  booktitle = {Audio Engineering Society Convention 147},
  address   = {New York, NY, USA},
  year      = {2019},
  month     = oct,
  note      = {Paper 10268}
}

@inproceedings{mourgela2024trends,
  author    = {Mourgela, Angeliki and Quinton, Elio and Bissas, Spyridon and Reiss, Joshua D. and Ronan, David},
  title     = {Exploring trends in audio mixes and masters: Insights from a dataset analysis},
  booktitle = {Audio Engineering Society Convention 157},
  address   = {New York, NY, USA},
  year      = {2024},
  month     = oct,
  note      = {Paper 10186. arXiv:2412.03373}
}

@inproceedings{shepherd2017psr,
  author    = {Shepherd, Ian and Grimm, Eelco and Tapper, Paul and Kahsnitz, Michael and Kerr, Ian},
  title     = {Measuring Micro-Dynamics: A First Step: Standardizing {PSR}, the Peak to Short-Term Loudness Ratio},
  booktitle = {Audio Engineering Society Convention 143},
  address   = {New York, NY, USA},
  year      = {2017},
  month     = oct,
  note      = {Engineering Brief 373. Author order to be confirmed against the AES E-Library record.}
}

@inproceedings{nielsen2000zerodbfs,
  author    = {Nielsen, S{\o}ren H. and Lund, Thomas},
  title     = {{0dBFS+} Levels in Digital Mastering},
  booktitle = {Audio Engineering Society Convention 109},
  address   = {Los Angeles, CA, USA},
  year      = {2000},
  month     = sep,
  note      = {Preprint 5251}
}

@inproceedings{lund2006stop,
  author    = {Lund, Thomas},
  title     = {Stop Counting Samples},
  booktitle = {Audio Engineering Society Convention 121},
  address   = {San Francisco, CA, USA},
  year      = {2006},
  month     = oct,
  note      = {Paper 6972}
}

% ---------- Books ----------

@book{katz2014mastering,
  author    = {Katz, Bob},
  title     = {Mastering Audio: The Art and the Science},
  edition   = {3},
  publisher = {Focal Press},
  year      = {2014},
  isbn      = {9780240818962}
}

% ---------- Practitioner articles, guides and interviews ----------

@misc{izotope2015guide,
  author       = {{iZotope, Inc.}},
  title        = {Mastering with {Ozone}: Tools, Tips and Techniques},
  year         = {2015},
  howpublished = {Educational guide, 2015 edition revised by Jonathan Wyner},
  url          = {https://downloads.izotope.com/guides/iZotopeMasteringGuide_MasteringWithOzone.pdf},
  note         = {Accessed 2026-10-02}
}

@misc{galindo2022eq,
  author       = {Galindo, Jett},
  title        = {Advanced {EQ} mastering tips},
  year         = {2022},
  month        = sep,
  howpublished = {iZotope Learn},
  url          = {https://www.izotope.com/en/learn/advanced-eq-mastering-tips},
  note         = {Accessed 2026-10-02}
}

@misc{galindo2022compression,
  author       = {Galindo, Jett},
  title        = {How to use compression in mastering},
  year         = {2022},
  month        = sep,
  howpublished = {iZotope Learn},
  url          = {https://www.izotope.com/en/learn/how-to-use-compression-in-mastering},
  note         = {Accessed 2026-10-02}
}

@misc{luthar2023stems,
  author       = {Luthar, Margaret},
  title        = {Stem mastering: How and when to use stems in audio mastering},
  year         = {2023},
  month        = sep,
  howpublished = {iZotope Learn},
  url          = {https://www.izotope.com/en/learn/stem-mastering},
  note         = {Accessed 2026-10-02}
}

@misc{deruty2011sos,
  author       = {Deruty, Emmanuel},
  title        = {'Dynamic Range' \& The Loudness War},
  year         = {2011},
  month        = sep,
  howpublished = {Sound on Sound},
  url          = {https://www.soundonsound.com/sound-advice/dynamic-range-loudness-war},
  note         = {Accessed 2026-10-02}
}

@misc{james2015vinyl,
  author       = {James, Eric},
  title        = {Q. How does mastering differ for vinyl and digital releases?},
  year         = {2015},
  month        = mar,
  howpublished = {Sound on Sound},
  url          = {https://www.soundonsound.com/sound-advice/q-how-does-mastering-differ-vinyl-and-digital-releases},
  note         = {Accessed 2026-10-02}
}

@misc{robjohns2016correlation,
  author       = {Robjohns, Hugh},
  title        = {Q. What are my phase-correlation meters telling me?},
  year         = {2016},
  month        = oct,
  howpublished = {Sound on Sound},
  url          = {https://www.soundonsound.com/node/4914566},
  note         = {Accessed 2026-10-02}
}

@misc{senior2009deessing,
  author       = {Senior, Mike},
  title        = {Techniques For Vocal De-essing},
  year         = {2009},
  month        = may,
  howpublished = {Sound on Sound},
  url          = {https://www.soundonsound.com/techniques/techniques-vocal-de-essing},
  note         = {Accessed 2026-10-02}
}

@misc{wright2025mastering,
  author       = {Wright, Alexander},
  title        = {What Mastering Can \& Can't Do},
  year         = {2025},
  month        = sep,
  howpublished = {Sound on Sound},
  url          = {https://www.soundonsound.com/techniques/what-mastering-can-cant-do},
  note         = {Accessed 2026-10-02}
}

@misc{owsinski2016trouble,
  author       = {Owsinski, Bobby},
  title        = {In The Studio: The 6 Trouble Frequencies},
  year         = {2016},
  month        = may,
  howpublished = {ProSoundWeb},
  url          = {https://www.prosoundweb.com/in-the-studio-the-6-trouble-frequencies/},
  note         = {Accessed 2026-10-02}
}

@misc{productionexpert2025clippers,
  author       = {{Production Expert}},
  title        = {What Are Clipper Plugins And Why Use Them?},
  year         = {2025},
  month        = feb,
  url          = {https://www.production-expert.com/production-expert-1/what-are-clipper-plugins-and-why-use-them},
  note         = {Accessed 2026-10-02}
}

@misc{fabfilter_prol2,
  author       = {{FabFilter}},
  title        = {{FabFilter Pro-L 2} Help: True peak limiting; Advanced settings},
  url          = {https://www.fabfilter.com/help/pro-l/using/truepeaklimiting},
  note         = {Accessed 2026-10-02}
}

@misc{shepherd_dynameter,
  author       = {Shepherd, Ian and {MeterPlugs}},
  title        = {Ian Shepherd's Dynameter: User Manual},
  url          = {https://www.meterplugs.com/files/dynameter-guide.pdf},
  note         = {Accessed 2026-10-02}
}

@misc{meterplugs2017psr,
  author       = {{MeterPlugs}},
  title        = {Crest Factor, {PSR} and {PLR}},
  year         = {2017},
  month        = may,
  url          = {https://www.meterplugs.com/blog/2017/05/18/crest-factor-psr-and-plr.html},
  note         = {Accessed 2026-10-02}
}

@misc{meterplugs2019youtube,
  author       = {{MeterPlugs}},
  title        = {{YouTube} Changes Loudness Reference to -14 {LUFS}},
  year         = {2019},
  month        = sep,
  url          = {https://www.meterplugs.com/blog/2019/09/18/youtube-changes-loudness-reference-to-14-lufs.html},
  note         = {Accessed 2026-10-02}
}

@misc{meterplugs2019amazon,
  author       = {{MeterPlugs}},
  title        = {Loudness Penalty Now Supports {Amazon Music}, {Deezer}},
  year         = {2019},
  month        = oct,
  url          = {https://www.meterplugs.com/blog/2019/10/15/loudness-penalty-amazon-deezer.html},
  note         = {Accessed 2026-10-02}
}

@misc{meterplugs2022apple,
  author       = {{MeterPlugs}},
  title        = {{Apple} Switches to {LUFS}, Enables Sound Check by Default},
  year         = {2022},
  month        = mar,
  url          = {https://www.meterplugs.com/blog/2022/03/23/apple-switch-to-lufs.html},
  note         = {Accessed 2026-10-02}
}

@misc{audioxpress2017tidal,
  author       = {{audioXpress}},
  title        = {{Tidal} Implements Album Loudness Normalization and Activates It by Default for Mobile Players},
  year         = {2017},
  url          = {https://audioxpress.com/news/tidal-implements-album-loudness-normalization-and-activates-it-by-default-for-mobile-players},
  note         = {Accessed 2026-10-02}
}

@misc{drdb_faq,
  author       = {{Dynamic Range DB}},
  title        = {{FAQ}},
  url          = {https://dr.loudness-war.info/faq},
  note         = {Accessed 2026-10-02}
}

@misc{ludwig2011castle,
  author       = {Castle, Chris and Ludwig, Bob},
  title        = {Recording Tips for the 'The Loudness Wars': An Interview With Mastering Great {Bob Ludwig}},
  year         = {2011},
  month        = dec,
  url          = {https://www.christiancastle.com/articles/2014/1/16/recording-tips-for-the-the-loudness-wars-an-interview-with-mastering-great-bob-ludwig},
  note         = {Accessed 2026-10-02}
}

@misc{ludwig2016mc,
  author       = {Putnam, Rob and Ludwig, Bob},
  title        = {Music Industry Advice: Mastering Insights from Experts},
  year         = {2016},
  month        = oct,
  howpublished = {Music Connection},
  url          = {https://www.musicconnection.com/?p=57823},
  note         = {Accessed 2026-10-02}
}

% ---------- Bob Katz, Digital Domain (digido.com) ----------

@misc{katz_levelpractices2,
  author       = {Katz, Bob},
  title        = {Level Practices (Part 2)},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/portfolio-item/level-practices-part-2/},
  note         = {Accessed 2026-10-02. Web article presenting the K-System described in the 2000 JAES paper.}
}

@misc{katz_compression_article,
  author       = {Katz, Bob},
  title        = {Compression},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/portfolio-item/compression/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_dither_article,
  author       = {Katz, Bob},
  title        = {Dither},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/portfolio-item/dither/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_audio_mastering,
  author       = {Katz, Bob},
  title        = {Audio Mastering},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/portfolio-item/audio-mastering/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_dither,
  author       = {Katz, Bob},
  title        = {{FAQ}: Dither, when to?},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/dither-when-to/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_src,
  author       = {Katz, Bob},
  title        = {{FAQ}: Sample Rate Conversion ({SRC}) or {EQ}, which comes first?},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/sample-rate-conversion-src-or-eq-which-comes-first/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_gr,
  author       = {Katz, Bob},
  title        = {{FAQ}: Compression: How much gain reduction to use? Pumping, how to avoid? Limiting versus compression?},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/compression-much-gain-reduction-use-pumping-avoid-limiting-versus-compression/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_dc,
  author       = {Katz, Bob},
  title        = {{FAQ}: {DC} Offset: Clicks and Pops},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/dc-offset-clicks-pops/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_removing,
  author       = {Katz, Bob},
  title        = {{FAQ}: Removing Frequencies in Mastering},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/removing-frequencies-in-mastering/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_bass,
  author       = {Katz, Bob},
  title        = {{FAQ}: Bass Problems},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/bass-problems/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_bassheavy,
  author       = {Katz, Bob},
  title        = {{FAQ}: Bass-heavy or Bass-light Mixes: Which is easier to master?},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/bass-heavy-bass-light-mixes-easier-master/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_ms,
  author       = {Katz, Bob},
  title        = {{FAQ}: {MS} {EQ}},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/ms-eq/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_linphase,
  author       = {Katz, Bob},
  title        = {{FAQ}: Linear Phase Equalization},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/linear-phase-equalization/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_truepeak,
  author       = {Katz, Bob},
  title        = {{FAQ}: True Peak Metering},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/true-peak-metering/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_noise,
  author       = {Katz, Bob},
  title        = {{FAQ}: Noise},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/noise/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_dance,
  author       = {Katz, Bob},
  title        = {{FAQ}: Compression in Mastering Dance Music},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/compression-mastering-dance-music/},
  note         = {Accessed 2026-10-02}
}

@misc{katz_faq_eqfinal,
  author       = {Katz, Bob},
  title        = {{FAQ}: {EQing} the Final Mix},
  howpublished = {Digital Domain},
  url          = {https://www.digido.com/ufaqs/eqing-final-mix/},
  note         = {Accessed 2026-10-02}
}
```
