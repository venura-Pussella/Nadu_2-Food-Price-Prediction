def create_sequences(df, sequence_length=30):
    sequences = []
    real_values = []
    dates = []
    
    for i in range(len(df) - sequence_length):
        # pass pandas Series directly
        sequence = df['pettah_average'].iloc[i:i + sequence_length] 

        # The real value is the 31st value
        real_value = df['pettah_average'].iloc[i + sequence_length]
        
        # The date associated with the real value
        real_date = df['date'].iloc[i + sequence_length]
        
        # Append the sequence, real values, and date
        sequences.append(sequence)
        real_values.append(real_value)
        dates.append(real_date)
    
    return sequences, real_values, dates