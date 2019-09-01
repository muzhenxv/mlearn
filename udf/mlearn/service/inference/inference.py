from ..data_service import get_data

def filter_inference(df_src, label, enc):
    df = get_data(df_src)
    label = df.pop(label)

