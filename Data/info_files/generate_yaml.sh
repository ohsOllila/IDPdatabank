#!/bin/bash



for dir in *; do
    if [[ -d "$dir" && "$dir" =~ ^(18[1-9]|19[0-9]|20[0-5])$ ]]; then
        echo "Entering $dir"
        cd "$dir"

        yaml_file=$(ls *.yaml 2>/dev/null)

        if [ -n "$yaml_file" ]; then
            echo "Running script with $yaml_file"
            PYTHONPATH=/home/cmcajsa/Documents/IDPdatabank/Scripts python /home/cmcajsa/Documents/IDPdatabank/Scripts/BuildDatabank/AddData.py -f "$yaml_file"
        else
            echo "No YAML file found in $dir"
        fi

        cd ..
    fi
done
