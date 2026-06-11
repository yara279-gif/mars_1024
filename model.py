from models.MARs import MARs

# Load MARs model
def set_model(conformer_model_dim=256,
              conformer_nhead=4,
              conformer_num_blocks=4,
              conformer_conv_kernel_sizes=[31],
              conformer_dropout=0.1,
              conformer_order='att_first'):
    model = MARs(
        input_dim=1024,  # GoogleNet pool5 feature dimension
        model_dim=conformer_model_dim,
        nhead=conformer_nhead,
        num_blocks=conformer_num_blocks,
        conv_kernel_sizes=conformer_conv_kernel_sizes,
        dropout=conformer_dropout,
        order=conformer_order
    )
    return model