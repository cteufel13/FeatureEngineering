def drop_db_cols(data):

    return data.drop(["rtype", "publisher_id", "instrument_id", "symbol"], axis=1)
