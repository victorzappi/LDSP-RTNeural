/* 
    Implementation of the neural model descrinbd im:
    Simionato, Riccardo, and Stefano Fasciani. "Fully Conditioned and Low-latency Black-box Modeling of Analog Compression." In Proceedings of the International Conference on Digital Audio Effects. DAFx Board, 2023.
    https://github.com/RiccardoVib/CONDITIONED-MODELING-OF-OPTICAL-COMPRESSOR
*/

#pragma once

#include <RTNeural/RTNeural.h>
#include "lstm_eigen_custom.h"

using json = nlohmann::json;

template <std::size_t w, std::size_t u, std::size_t d>
class RT_ED
{
public:
    RT_ED();

    void reset();
    void load_json(const char* filename);
    void set_weights(const json& weights_json);

    void process(float* inData, float* condParams, float* outData);

private:
    // Encoder dense layers for conditional input data 
    RTNeural::ModelT<float, d, u,
        RTNeural::DenseT<float, d, u>> encoder_dense_h; // Dense layer for state_h
    RTNeural::ModelT<float, d, u,
        RTNeural::DenseT<float, d, u>> encoder_dense_c; // Dense layer for state_c

    // Encoder convolutional layers for past input data
    RTNeural::ModelT<float, 1, u,
        RTNeural::Conv1DT<float, /*in_sizet=*/1, /*out_sizet(filter num)=*/u, /*kernel_size=*/w, /*dilation_rate =*/1>> encoder_conv_h; // Convolutional layer for state_h
    RTNeural::ModelT<float, 1, u,
        RTNeural::Conv1DT<float, /*in_sizet=*/1, /*out_sizet(filter num)=*/u, /*kernel_size=*/w, /*dilation_rate =*/1>> encoder_conv_c; // Convolutional layer for state_c

    // Decoder which will receive the concatenated states from the encoder
    RTNeural::ModelT<float, 1, u,
        RTNeural::CustomLSTMLayerT<float, 1, u>> decoder_lstm; // LSTM layer
    RTNeural::ModelT<float, u, w,
        RTNeural::DenseT<float, u, u>, // Additional Dense layer before activation, if needed
        RTNeural::SigmoidActivationT<float, u>, // Activation layer
        RTNeural::DenseT<float, u, w>> decoder_dense; // Final dense output layer

    void loadCustomLSTM(RTNeural::CustomLSTMLayerT<float, 1, u>& lstm, const nlohmann::json& weights);

    alignas(16) float condParams[d];
    alignas(16) float inDataBlock[2*w];
    alignas(16) float states_h[u];
    alignas(16) float states_c[u];
    Eigen::Matrix<float, u, 1> eigen_states_h;
    Eigen::Matrix<float, u, 1> eigen_states_c;
};



template <std::size_t w, std::size_t u, std::size_t d>
RT_ED<w, u, d>::RT_ED()
{
    eigen_states_h = Eigen::MatrixXf::Zero(u, 1);
    eigen_states_c = Eigen::MatrixXf::Zero(u, 1);
}

template <std::size_t w, std::size_t u, std::size_t d>
void RT_ED<w, u, d>::reset()
{
   encoder_dense_h.reset();
   encoder_dense_c.reset();

   encoder_conv_h.reset();
   encoder_conv_c.reset();

   decoder_lstm.reset();
   decoder_dense.reset();
}

template <std::size_t w, std::size_t u, std::size_t d>
void RT_ED<w, u, d>::load_json(const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open the weights file.");

    json weights_json;
    file >> weights_json;

    set_weights(weights_json);
}

template <std::size_t w, std::size_t u, std::size_t d>
void RT_ED<w, u, d>::set_weights(const json& parent)
{
    const int enc_dense_h_id = 3;
    const int enc_dense_c_id = 5;
    const int enc_conv_h_id = 0;
    const int enc_conv_c_id = 1;
    const int dec_lstm_id = 8;
    const int dec_dense_sig_id = 9;
    const int dec_dense_out_id = 10;

    auto layers = parent.at("layers");
    // Set weights for encoder_dense_h
    {        
        const auto weights = layers[enc_dense_h_id].at("weights");
        auto& dense = encoder_dense_h.template get<0>();
        RTNeural::json_parser::loadDense<float>(dense, weights);
    }

    // Set weights for encoder_dense_c
    {
        const auto weights = layers[enc_dense_c_id].at("weights");
        auto& dense = encoder_dense_c.template get<0>();
        RTNeural::json_parser::loadDense<float>(dense, weights);
    }

    // Set weights for encoder_conv_h
    {
        const auto weights = layers[enc_conv_h_id].at("weights");
        auto& conv = encoder_conv_h.template get<0>();
        RTNeural::json_parser::loadConv1D<float>(conv, w, 1, weights);
    }

    // Set weights for encoder_conv_c
    {
        const auto weights = layers[enc_conv_c_id].at("weights");
        auto& conv = encoder_conv_c.template get<0>();
        RTNeural::json_parser::loadConv1D<float>(conv, w, 1, weights);
    }

    // Set weights for decoder LSTM layer
    {
        const auto weights = layers[dec_lstm_id].at("weights");
        auto& lstm = static_cast<RTNeural::CustomLSTMLayerT<float, 1, u>&>(decoder_lstm.template get<0>());
        loadCustomLSTM(lstm, weights);
    }

    // Set weights for decoder dense layers
    // (No weights are needed for the sigmoid activation)
    {
        const auto weights = layers[dec_dense_sig_id].at("weights");
        auto& dense = decoder_dense.template get<0>();
        RTNeural::json_parser::loadDense<float>(dense, weights);
    }
    

    // Set weights for the final output dense layer
    {        
        const auto weights = layers[dec_dense_out_id].at("weights");
        auto& dense = decoder_dense.template get<2>();
        RTNeural::json_parser::loadDense<float>(dense, weights);
    }
}

/** Loads weights for a CustomLSTMLayerT from a json representation of the layer weights. */
template <std::size_t w, std::size_t u, std::size_t d>
inline void RT_ED<w, u, d>::loadCustomLSTM(RTNeural::CustomLSTMLayerT<float, 1, u>& lstm, const nlohmann::json& weights)
{
    // load kernel weights
    std::vector<std::vector<float>> kernelWeights(lstm.in_size);
    for(auto& kw : kernelWeights)
        kw.resize(4 * lstm.out_size, (float)0);

    auto layerWeights = weights.at(0);
    for(size_t i = 0; i < layerWeights.size(); ++i)
    {
        auto lw = layerWeights.at(i);
        for(size_t j = 0; j < lw.size(); ++j)
            kernelWeights.at(i).at(j) = lw.at(j).get<float>();
    }

    // Now set the kernel weights
    lstm.setWVals(kernelWeights);
    // load recurrent weights
    std::vector<std::vector<float>> recurrentWeights(lstm.out_size);
    for(auto& r : recurrentWeights)
        r.resize(4 * lstm.out_size, (float)0);

    auto layerWeights2 = weights.at(1);
    for(size_t i = 0; i < layerWeights2.size(); ++i)
    {
        auto lw = layerWeights2.at(i);
        for(size_t j = 0; j < lw.size(); ++j)
            recurrentWeights.at(i).at(j) = lw.at(j).get<float>();
    }

    // Now set the recurrent weights
    lstm.setUVals(layerWeights2);

    // load biases
    std::vector<float> lstmBias = weights.at(2).get<std::vector<float>>();

    // Now set the biases
    lstm.setBVals(lstmBias);
}


template <std::size_t w, std::size_t u, std::size_t d>
inline void RT_ED<w, u, d>::process(float* inData, float* params, float* outData)
{    
    for(int i=0; i<d; i++)
        condParams[i] = params[i];

    // Process condParams through encoder dense layers
    encoder_dense_h.forward(condParams);
    const float *cond_dense_h = static_cast<const float*>( encoder_dense_h.getOutputs() );
    encoder_dense_c.forward(condParams);
    const float *cond_dense_c = static_cast<const float*>( encoder_dense_c.getOutputs() );


    std::copy(inData, inData + 2*w, inDataBlock);

 
    // Process block of least recent w samples through encoder convolutional layers
    // encoder_conv_h.forward(/* inDataBlock+i */ inDataBlock);
    for(int i = 0; i < w; i++)
        encoder_conv_h.forward(&inDataBlock[i]);
    const float * conv_h = static_cast<const float*>( encoder_conv_h.getOutputs() );
    // encoder_conv_c.forward(/* inDataBlock+i */ inDataBlock);
    for(int i = 0; i < w; i++)
        encoder_conv_c.forward(&inDataBlock[i]);
    const float * conv_c = static_cast<const float*>( encoder_conv_c.getOutputs() );

    
    // Prepare LSTM states input by combining convolutional and dense outputs
    for (size_t s = 0; s < u; ++s) 
    {
        states_h[s] = conv_h[s] + cond_dense_h[s];
        states_c[s] = conv_c[s] + cond_dense_c[s]; 
    }


    // Set the states in the decoder LSTM
    auto& lstmLayer = static_cast<RTNeural::CustomLSTMLayerT<float, 1, u>&>(decoder_lstm.template get<0>());
    std::copy(states_h, states_h + u, eigen_states_h.data());
    std::copy(states_c, states_c + u, eigen_states_c.data());
    lstmLayer.setInitialHiddenState(eigen_states_h);
    lstmLayer.setInitialCellState(eigen_states_c);



    // Process block of most recent w samples through the decoder LSTM and dense layers
    for (int i = 0; i < w; i++) 
        decoder_lstm.forward(&inDataBlock[w+i]);
    const float* lstm_output = static_cast<const float*>( decoder_lstm.getOutputs() ); 

    decoder_dense.forward(lstm_output);
    const float* output = static_cast<const float*>( decoder_dense.getOutputs() );
    std::copy(output, output + w, outData);
 }

