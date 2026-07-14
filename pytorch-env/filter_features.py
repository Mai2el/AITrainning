
import pandas as pd

# Load dataset (CSV file)
input_file = 'merged_CSVs.csv'   # change to your file
output_file = 'filtered.csv'

df = pd.read_csv(input_file)

selected_features = ['duration', 'packets_count', 'fwd_packets_count', 'bwd_packets_count', 'total_header_bytes', 'max_header_bytes', 'min_header_bytes', 'mean_header_bytes', 'std_header_bytes', 'fwd_total_header_bytes', 'fwd_max_header_bytes', 'fwd_min_header_bytes', 'fwd_mean_header_bytes', 'fwd_std_header_bytes', 'bwd_total_header_bytes', 'bwd_max_header_bytes', 'bwd_min_header_bytes', 'bwd_mean_header_bytes', 'bwd_std_header_bytes', 'fwd_avg_segment_size', 'bwd_avg_segment_size', 'avg_segment_size', 'fwd_init_win_bytes', 'bwd_init_win_bytes', 'bytes_rate', 'fwd_bytes_rate', 'bwd_bytes_rate', 'packets_rate', 'bwd_packets_rate', 'fwd_packets_rate', 'down_up_rate', 'fin_flag_counts', 'psh_flag_counts', 'urg_flag_counts', 'syn_flag_counts', 'ack_flag_counts', 'rst_flag_counts', 'fin_flag_percentage_in_total', 'syn_flag_percentage_in_total', 'ack_flag_percentage_in_total', 'rst_flag_percentage_in_total', 'packets_IAT_mean', 'packet_IAT_std', 'packet_IAT_max', 'packet_IAT_min', 'delta_start', 'handshake_duration', 'handshake_state', 'mean_packets_delta_time', 'variance_packets_delta_time', 'std_packets_delta_time', 'mean_fwd_packets_delta_time', 'mode_fwd_packets_delta_time', 'label', 'activity']

# Keep only available columns (avoid errors)
available_features = [col for col in selected_features if col in df.columns]

filtered_df = df[available_features]

# Save filtered dataset
filtered_df.to_csv(output_file, index=False)

print(f'Filtered dataset saved to {output_file}')
