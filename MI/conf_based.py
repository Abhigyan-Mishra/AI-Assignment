import argparse
import os
from attacker_models.confidence_based_attack import confidence_attack


def check_known_dataset_size(dataset):
    if dataset < 0.1 or dataset > 0.5:
        print("Range should be in b/w [0.1 - 0.5]")
        exit()

    return True


def save_histogram(data):
    dir = os.path.join(os.getcwd(), f"figures/histogram/{data}")
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Attack based on confidence values")
    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist"], help="Dataset and Trained Model")
    parser.add_argument("-m", "--target_model", type=str,
                        default="shadow_models/mninst/mnist.h5", help="Path for shadow model")
    parser.add_argument("-a", "--attack_model", type=str, default="XGBoost",
                        choices=["NN", "XGBoost"], help="Defaulted to XGBoost for Attacker Classifier")
    parser.add_argument("-s", "--sampling", type=str, default=None,
                        choices=["none", "undersampling", "oversampling"])
    parser.add_argument("-c", "--knowledge", type=float,
                        default=0.5, help="Data available to the attacker.")
    parser.add_argument("-n", "--no_of_target_classes", type=int,
                        default=0, help="Number of classes")
    parser.add_argument('--save_hist', default=False,
                        help='Save confidence histogram of each class.', action='store_true')
    parser.add_argument('--show_separate_results', default=False,
                        help='Show results for correctly classified and misclassified samples, separately.', action='store_true')
    parser.add_argument('--verbose', default=False,
                        help='Print full details.', action='store_true')
    args = parser.parse_args()

    verbose = args.verbose
    data = args.dataset
    shadow_model = args.target_model
    attacker_classifier = args.attack_model
    sampling = args.sampling
    known_dataset = args.knowledge

    if check_known_dataset_size(known_dataset):
        conf_histogram = args.save_hist

        if conf_histogram:
            save_histogram(data)

        show_separate_results = args.show_separate_results
        save_dir = os.path.join(os.getcwd(), "saved_models")

        num_classes = 10
        if data == "mnist":
            num_targeted_classes = 10

        else:
            print("Unknown dataset!!")
            exit()

        confidence_attack(data, attacker_classifier, sampling, known_dataset, conf_histogram,
               show_separate_results,  num_classes, num_targeted_classes, shadow_model, verbose)
