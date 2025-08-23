from astronomical_prediction_model_data_process import generate
from astronomical_prediction_model import predict

if __name__ == '__main__':
    generate(start_year=450,
             start_month=1,
             start_day=1,
             end_year=500,
             end_month=12,
             end_day=1)

    predict(n=300)

