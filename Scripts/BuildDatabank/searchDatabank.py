from experiment import Experiment
from simulation import Simulation
import glob
import os
from databank_general_utils import setup_colored_logging, setup_colored_warnings
from build_databank_utils import align_and_evaluate_protein_sequences
from parameter_comparator import ParameterComparator, Threshold, ThresholdType


# Set up colored logging
logger = setup_colored_logging(__name__)

# Configure warnings to use colored output
setup_colored_warnings()

experiment_types = ["spin_relaxation"]  # will be extended


def main():
    comparator = ParameterComparator()
    IDP_DATABANK_PATH = "Data"
    IDP_DATABANK_EXPERIMENTS_PATH = os.path.join(IDP_DATABANK_PATH, "Experiments")
    IDP_DATABANK_SIMULATIONS_PATH = os.path.join(IDP_DATABANK_PATH, "Simulations")

    simulation_readmes = glob.glob(
        os.path.join(IDP_DATABANK_SIMULATIONS_PATH, "**/README.yaml"),
        recursive=True,
    )
    logger.info(f"Simulation readmes:\n {simulation_readmes}")

    # for now just look at the spin relaxation experiments
    experiment_readmes = glob.glob(
        os.path.join(IDP_DATABANK_EXPERIMENTS_PATH, "**/spin_relaxation_times.yaml"),
        recursive=True,
    )
    logger.info(f"Experiment readmes:\n {experiment_readmes}")

    simulations = []

    for simulation_readme in simulation_readmes:
        simulation = Simulation(simulation_readme)
        simulations.append(simulation)

    # TODO: extend the experiments beyond the spin_relaxation
    experiments = []
    experiment_types = ["spin_relaxation"]
    for experiment_readme in experiment_readmes:
        experiment_readme_dir = os.path.dirname(experiment_readme)
        experiment = Experiment(
            experiment_type="spin_relaxation", path=experiment_readme_dir
        )
        if experiment.metadata == {}:
            logger.warning(
                f"Experiment metadata for {experiment.path} is empty. Cannot match"
            )
            continue
        else:
            experiments.append(experiment)

    for simulation in simulations:

        # TODO: will need to loop through all the experiment types, not just spin_relaxation
        experiment_types_dict = {}

        simulation_ph = 7  # TODO: Implement pH-getter for the simulations
        simulation_temperature = simulation.info["TEMPERATURE"]
        simulation_ionic_strength = simulation.ionic_strength
        for experiment_type in experiment_types:

            experiment_types_dict[experiment_type] = {}
            experiment_types_dict[experiment_type]["path"] = []
            experiment_types_dict[experiment_type]["ph"] = []
            experiment_types_dict[experiment_type]["temperature"] = []
            experiment_types_dict[experiment_type]["ionic_strength"] = []
            experiment_types_dict[experiment_type]["sequence"] = []
            experiment_types_dict[experiment_type]["ph_match"] = []
            experiment_types_dict[experiment_type]["temperature_match"] = []
            experiment_types_dict[experiment_type]["ionic_strength_match"] = []
            experiment_types_dict[experiment_type]["alignment_score"] = []

        for experiment in experiments:

            # we first check for the exact sequence match
            # if not, we do an alignment, spit out alignment score. the exact threshold will
            # be determined later.
            # then, we put the ph, temperature, and the ionic strength of both the experiment and
            # simulation to the resulting README. we can consider putting "match temperature = True" kinda fields
            # to the README.

            sequence_match = simulation.sequence == experiment.sequence
            if sequence_match:
                logger.info(
                    f"Perfect sequence match found between simulation {simulation.path} and experiment {experiment.path}"
                )
                alignment_score = -1  # unique value for exact match
            else:
                logger.info(
                    f"sequence_1: {simulation.sequence}, sequence_2: {experiment.sequence}"
                )
                alignment = align_and_evaluate_protein_sequences(
                    simulation.sequence, experiment.sequence
                )

                hypothetical_alignment_score = align_and_evaluate_protein_sequences(
                    simulation.sequence, simulation.sequence
                ).score

                alignment_score_threshold = (
                    hypothetical_alignment_score * 0.8
                )  # this is as arbitrary as it gets
                if alignment is not None:
                    if alignment.score > alignment_score_threshold:
                        alignment_score = alignment.score
                        logger.info(
                            f"Alignment found between simulation {simulation.path} and experiment {experiment.path} with score {alignment.score}"
                        )
                    else:
                        logger.info(
                            f"Alignment found between simulation {simulation.path} and experiment {experiment.path} with score {alignment.score} but below threshold {alignment_score_threshold}."
                        )
                        continue
                else:
                    logger.info(
                        f"No alignment found between simulation {simulation.path} and experiment {experiment.path}"
                    )
                    continue

            # matching_experiments.append(experiment)
            experiment_types_dict[experiment_type]["path"].append(
                os.path.basename(experiment.path)
            )
            experiment_ph = experiment.ph
            experiment_types_dict[experiment_type]["ph"].append(experiment_ph)

            experiment_temperature = experiment.temperature
            experiment_types_dict[experiment_type]["temperature"].append(
                experiment_temperature
            )

            experiment_ionic_strength = experiment.ionic_strength
            experiment_types_dict[experiment_type]["ionic_strength"].append(
                experiment_ionic_strength
            )

            experiment_sequence = experiment.sequence
            experiment_types_dict[experiment_type]["sequence"].append(
                experiment_sequence
            )

            #
            ph_result = comparator.compare("ph", simulation_ph, experiment_ph)
            temperature_result = comparator.compare(
                "temperature", simulation_temperature, experiment_temperature
            )
            ionic_result = comparator.compare(
                "ionic_strength", simulation_ionic_strength, experiment_ionic_strength
            )

            # Update your dictionary with the results
            experiment_types_dict[experiment_type]["ph_match"].append(ph_result.match)
            experiment_types_dict[experiment_type]["temperature_match"].append(
                temperature_result.match
            )
            experiment_types_dict[experiment_type]["ionic_strength_match"].append(
                ionic_result.match
            )

            experiment_types_dict[experiment_type]["alignment_score"].append(
                alignment_score
            )

        logger.info(
            f"Would append this to the info file \n{simulation.path}\n {experiment_types_dict}"
        )

        simulation.info["EXPERIMENT"] = experiment_types_dict

        temp_path = simulation.path.with_name("README_matched.yaml")
        simulation.file_handler.write_yaml(temp_path, simulation.info)


if __name__ == "__main__":
    main()
