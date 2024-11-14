from constants import *


def format_dataset(dataset, filename):
    # processing na fields
    for column in [LAND_CONTOUR_KEY, GARAGE_YR_BLT_KEY, FULL_BATH_KEY, TOT_RMS_ABV_GRD_KEY,
                   EXTER_QUAL_KEY, HEATING_KEY, CONDITION_2_KEY, GARAGE_CARS_KEY, KITCHEN_ABV_GR_KEY,
                   OVERALL_QUAL_KEY, KITCHEN_QUAL_KEY, CENTRAL_AIR_KEY, BSMT_QUAL_KEY, FIREPLACES_KEY]:
        dataset.loc[:, column] = dataset[column].fillna(dataset[column].mode()[0])

    for column in [FIRST_FLR_SF_KEY, BSMT_FIN_SF_1_KEY, OPEN_PORCH_SF_KEY,
                   GR_LIV_AREA_KEY, SECOND_FLR_SF_KEY, TOTAL_BSMT_SF_KEY]:
        dataset.loc[:, column] = dataset[column].fillna(dataset[column].mean())

    # deleting corr fields
    corr_fields = [GR_LIV_AREA_KEY, FIRST_FLR_SF_KEY]
    new_dataset = dataset.drop(corr_fields, axis=1)

    create_formatted_file(new_dataset, f'{filename}')


def create_formatted_file(df, filename):
    df.to_csv(f'{filename}', index=False, encoding='utf-8')


def create_result_file(df):
    df.to_csv(f'result/result.csv', index=False, encoding='utf-8')
