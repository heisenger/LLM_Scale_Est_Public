experiments/
  marker_location/
    runs/                                   # every acquisition “run” lives here
      20240804_text/
        llm=anthropic/
          raw/                              # immutable inputs
            trials.parquet                  # one row per trial: correct, response, range, trial, stimulus_id
            prompt.txt
          derived/                          # numbers/tables you can recompute
            overall.csv
            per_range.csv
            predictions.parquet
            ...
          reports/                          # figures & quick summaries
            evidence.png
            biasvar.png
          manifest.json

      20240804_image/
        llm=anthropic/
          raw/ ...                          # same columns as text
          derived/ ...
          reports/ ...
          manifest.json

      20240804_text_image/                  # the **combo raw** (LLM given both)
        llm=anthropic/
          raw/
            trials.parquet                  # same schema; response = LLM(Combined)
          derived/                          # (optional per-mod metrics)
          reports/
          manifest.json

    runsets/                                # where you do the *cue-combination* analysis
      20240804_comboA/                      # your chosen ID for this pairing
        manifest.json                       # references the 3 member runs
        llm=anthropic/
          derived/                          # outputs from cue_combination
            overall.csv
            per_range.csv
            predictions.parquet
            weights_bayes_per_range.csv
            weights_bayes_per_stimulus.csv
            weights_global.json
          reports/
            evidence.png
            biasvar.png
            weights.png
