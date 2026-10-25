# ECG Feature Classification Demo

**Explore how prepared ECG-derived features move through a machine-learning classification demo.** The Streamlit app accepts one feature record or a CSV of records and displays the class predicted by the bundled model.

[View the source](https://github.com/AhmedMagdi2003/Cardiac_Ailment)

> **Educational use only.** This project is not a medical device and has not been established as clinically validated. Do not use it for diagnosis, treatment, or urgent-care decisions.

## The project

The app provides an interactive way to explore a classifier using seven engineered feature values: NNTot, STdis, IBISD, PonPQang, STToffang, RRTot, and Pseg. Its output is one of four labels: Arrhythmia (ARR), Atrial Fibrillation & Flutter (AFF), Congestive Heart Failure (CHF), or Normal Sinus Rhythm (NSR).

## What you can explore

- Enter a prepared feature record in the Streamlit interface.
- Upload a CSV containing prepared feature rows for prediction.
- Review the model-development notebook and included project data.

## Try it locally

Install the packages listed in requirements.txt, keep ecg_model.pkl where deployment.py expects it, and launch the interface from the repository root:

~~~bash
streamlit run deployment.py
~~~

The app accepts derived features; the repository does not establish a raw ECG signal-processing pipeline. Confirm the feature-generation method, units, and model provenance before interpreting any output.

## Built with

Python · Streamlit · scikit-learn · pandas

## Scope and responsible use

The predictions reflect only the submitted feature values and the bundled estimator. The repository does not report clinical validation, calibrated risk estimates, or clinical safeguards. This is a software demonstration and should not influence real-world medical decisions.

## About

Built by Ahmed Magdi as an applied machine-learning demonstration. [Repository](https://github.com/AhmedMagdi2003/Cardiac_Ailment)
