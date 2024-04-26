#!/bin/bash

is_valid_number () {
    local var_name="$1"
    local var_value="${!var_name}"
    local re='^[0-9]+$'

    if [[ -z $var_value || ! $var_value =~ $re ]]; then
        echo 0
    else 
        echo 1
    fi
}

number_of_runs=""
number_of_rounds=""

optstring=":m:r:"

while getopts ${optstring} flag; do
    case "${flag}" in
    m) 
        number_of_runs=${OPTARG} 
        ;;
    r)
        number_of_rounds=${OPTARG} 
        ;;
    :)
        echo "option -${OPTARG} requires an argument."
        exit 1
        ;;
    ?) 
        echo "invalid option: -${OPTARG}."
        exit 1 
        ;;
    esac
done

if [[ $(is_valid_number number_of_runs) -eq 0 || $(is_valid_number number_of_rounds) -eq 0 ]]; then
    echo "ERROR: you must pass an integer value"
    echo "usage: fl_pipeline -n <number_of_models> -r <number_of_rounds>"
    exit 1
fi

relative_path="$(dirname "$0")"
cd "$relative_path/../../" || exit

temp_metrics_file_path="prototype_1/federated/metrics/temp_metrics.txt"
federated_path="prototype_1.federated"

simulation_file="simulation"
aggregating_metrics_file="aggregating_metrics"
simulation_module="${federated_path}.${simulation_file}"
aggregating_metrics_module="${federated_path}.${aggregating_metrics_file}"

if [ -f "$temp_metrics_file_path" ]; then
    rm "$temp_metrics_file_path"
fi

echo -e "\nTraining ${number_of_runs} federated learning models"
for (( i=1; i<=number_of_runs; i++ )); do
    echo -e "\nStarting to train model ${i}\n"
    if ! pgrep -f "${simulation_module}" > /dev/null; then
        python -m "${simulation_module}" --num-rounds "${number_of_rounds}" &
        wait $!
    fi
done

echo -e "\nTraining process of ${number_of_runs} federated models has finished."
echo "Starting to aggregate metrics into a json file..."

python -m "${aggregating_metrics_module}" --num-models "${number_of_runs}" --num-rounds "${number_of_rounds}" &
wait $!
