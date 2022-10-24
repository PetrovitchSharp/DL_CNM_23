import argparse
import joblib
import json


def make_parser() -> argparse.ArgumentParser:
    '''
    Make cli arguments parser

    Returns:
        CLI args parser
    '''
    parser = argparse.ArgumentParser(
        description='Inference for different models'
    )

    parser.add_argument('-input', type=str,
                        default='test.json',
                        help='json file with company names')

    parser.add_argument('-model', type=str,
                        default='uni_catboost_v0.joblib',
                        help='model filename')

    return parser

def main() -> None:
    '''
    Main function responsible for model inference
    '''

    # Arguments parsing
    parser = make_parser()
    args = parser.parse_args()

    input_file = args.input
    model_file = args.model

    # Model loading
    model = joblib.load(f'../../models/{model_file}')


    with open(input_file) as input:
        comp_names = json.load(input)

    company_name_1 = comp_names["company_1"]
    company_name_2 = comp_names["company_2"]

    prediction = model.predict(
        company_name_1,
        company_name_2
    )

    probabilities_array = model.predict_proba(
        company_name_1,
        company_name_2
    )

    # We choose probability of predicted class
    prob = probabilities_array[1] if prediction else probabilities_array[0]

    print(f'Company names are matched: {prediction} with probability: {prob*100}%')

if __name__ == '__main__':
    main()