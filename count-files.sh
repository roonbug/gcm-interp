#!/usr/bin/env bash
shopt -s globstar

root="/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results"
methods=(acp atp atp-zero probes random)

for variable in mean steer; do
    echo "################# ${variable} ################"

    # eval directory differs for pyreft
    if [[ ${variable} == "pyreft" ]]; then
        sub_dir="eval"
    else
        sub_dir="eval_test"
    fi

    unset counts
    declare -A counts

    # Collect counts
    for path in "$root"/**/${sub_dir}/*_*_${variable}_topk_*_gen.txt; do
        [[ -f "$path" ]] || continue

        if [[ "$path" =~ normalized-results/([^/]+)/from_([^_]*)_to_([^/]*)/([^/]+)/${sub_dir}/ ]]; then
            model="${BASH_REMATCH[1]}"
            source="${BASH_REMATCH[2]}"
            base="${BASH_REMATCH[3]}"
            method="${BASH_REMATCH[4]}"

            key="${model}:::${source}->${base}:::${method}"
            ((counts["$key"]++))
        fi
    done

    # --- Print outputs grouped by model ---
    # Identify all models that appear
    declare -A models_seen
    for k in "${!counts[@]}"; do
        model="${k%%:::*}"
        models_seen["$model"]=1
    done

    # Print per model
    for model in "${!models_seen[@]}"; do
        echo ""
        echo "== Model: ${model} =="

        # Extract all source->base pairs for this model
        declare -A pairs
        for k in "${!counts[@]}"; do
            m="${k%%:::*}"
            [[ "$m" == "$model" ]] || continue
            rest="${k#*:::}"        # source->base:::method
            pair="${rest%%:::*}"
            pairs["$pair"]=1
        done

        # Now print counts for each pair × each method
        for pair in "${!pairs[@]}"; do
            echo "  Pair: ${pair}"
            for method in "${methods[@]}"; do
                key="${model}:::${pair}:::${method}"
                c="${counts[$key]:-0}"
                printf "      %-10s %6d\n" "$method" "$c"
            done
        done
    done
done
