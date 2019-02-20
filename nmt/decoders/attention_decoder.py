from nmt.decoders.basic_decoder import BasicRNNDecoder


class AttentionDecoder(BasicRNNDecoder):

    def _build_cell_and_initial_state(self, encoder_output, encoder_state, params):
        pass

    def _build_decoder(self, cell, initial_state, params):
        pass

    def default_config(self):
        config = super(AttentionDecoder, self).default_config()
        config.update({
            "beam_width": 5,
            "attention": "luong",
            "attention_architecture": "standard",
            "sampling_temperature": 0.0,
        })
        return config
