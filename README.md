# MOSA: Music mOtion and Semantic Annotation dataset

![alt text](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/figure/dataset.png)



<p align="center">
  
https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/assets/42167635/13426b3c-1133-4a4d-96cd-860e6b772b1f

</p>




MOSA dataset is a large-scale music dataset containing 742 professional piano and violin solo music performances with 23 musicians (> 30 hours, and > 570 K notes). This dataset features following types of data
- High-quality 3-D motion capture data
- Audio recordings
- Manual semantic annotations


# 1. 3-D motion capture data
The 3-D motion capture data in this dataset are recorded using Qualisys and Vicon 3-D motion capture system. The original coordinates of 34 body/instrument markers on the x-, y-, and z- axes are provided. The body marker placement
follows the Plug-In Gait Full-body model in Visual-3D's official documention.

![alt text](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/figure/mocap.png)

# 2. Music semantic annotation

In MOSA dataset, we assemble high-quality manual-crafted semantic annotations for music performance. The statistics of annotations in MOSA dataset is shown in the Figure , and the collection of music annotations includes:

- Note annotations: pitch name (e.g. C4), midi number, note onset time, note offset time, note duration.
- Beat/downbeat annotations: the position of beat and downbeat, beat time, downbeat time.
- Harmony annotations: key (e.g. C major), Roman notation for harmony analysis, root degree (I - VII), inversion (root, 1st - 3rd inversion), chord quality (major triad (M), minor triad (m), augmented triad (A), diminished triad (d), dominant seventh (D7), major seventh (M7), minor seventh (m7),  diminished seventh (d7), half-diminished seventh (h7), augmented sixth (A6)),  chord onset time, chord offset time.
- Expressive annotations: dynamic marks (ppp - fff, crescendo, diminuendo, accent), tempo variations (accelerado, ritenuto, a tempo), and articulations (legato, staccato).
- Cadence annotations: high-level harmonic progression in music, cadence type (authentic (AC), half (HC), deceptive cadence (DC)), cadence onset time, cadence resolve time, cadence offset time.
- Structural annotations: phrase boundary, section boundary, motive, section type (e.g. exposition, development, recapitulation).

![alt text](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/figure/annot.png)

# 3. Audio recordings
