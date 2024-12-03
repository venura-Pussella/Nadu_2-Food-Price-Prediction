import pandas as pd

def validate_all_columns(config):

    try:
        validation_status = True  # Assume valid unless proven otherwise

        # Load data from the specified Excel file (explicitly specify the engine)
        data = pd.read_excel(config.unzip_data_dir)
        all_cols = list(data.columns)

        # Load schema keys
        all_schema = config.all_schema.keys()

        # Iterate through all columns and validate
        for col in all_cols:
            if col not in all_schema:
                validation_status = False
                with open(config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}\n")
                break  # Exit the loop if any column is invalid
        else:
            with open(config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")

        return validation_status

    except Exception as e:
        raise e