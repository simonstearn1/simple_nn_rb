# Simple neural network in ruby
#
#

require 'debug'
require_relative 'fish_exceptions'

module Fish

  class Layer < Array
  end

  class Weight < Array
  end

  class Net
    include Math

    attr_accessor :inputs, :hidden_nodes, :outputs, :learning_rate, :epochs, :debug, :data, :expected

    # Constants..
    DEFAULT_NUM_INPUTS = 2
    DEFAULT_NUM_HIDDEN_NODES = 2
    DEFAULT_NUM_OUTPUTS = 1
    DEFAULT_LEARNING_RATE = 0.1
    NUM_TRAINING_SETS = 4
    DEFAULT_NUM_EPOCHS = 100000

    def initialize(h = {})
      h.each {|k,v| public_send("#{k}=",v)}

      @hidden_layer = create_hidden_layer
      @output_layer = create_output_layer
      @output_layer_bias = create_output_layer_bias
      @hidden_weights = create_hidden_weights
      @hidden_layer_bias = create_hidden_layer_bias
      @output_weights = create_output_weights

      @training_set_order = training_set_initial_order

      @trained = false

    end

    def train

      (1..num_epochs).each do |epoch|
        puts "Epoch: #{epoch}." if @debug
        @training_set_order.shuffle! # This is Stochastic Gradient Decent so randomise training set order

        @training_set_order.each do |training_set|
          forwards_prop(training_set)

          puts "Input: #{training_inputs[training_set][0]} #{training_inputs[training_set][1]}  Output:#{@output_layer[0]} Expected: #{training_outputs[training_set][0]}" if @debug

          backwards_prop(training_set)
        end
      end

      if @debug
        puts "Final hidden weights: #{@hidden_weights}"
        puts "Final hidden biases: #{@hidden_layer_bias}"
        puts "Final output weights: #{@output_weights}"
        puts "Final output biases: #{@output_layer_bias}"
      end

      @trained = true

    end


    def predict(sample)
      raise NetworkNotTrained unless @trained
      raise SampleNotArray unless sample.is_a? Array
      raise SampleWrongLength unless sample.length == num_inputs
      raise SampleContainsNilValues if sample.include? nil

      forward_inputs(sample)
      forward_outputs

      return @output_layer
    end

    private

    # Constant wrappers - in case we want this to be smarter later..
    def num_inputs
      @inputs ||= DEFAULT_NUM_INPUTS
    end

    def num_hidden_nodes
      @hidden_nodes ||= DEFAULT_NUM_HIDDEN_NODES
    end

    def num_outputs
      @outputs ||= DEFAULT_NUM_OUTPUTS
    end

    def num_epochs
      @epochs ||= DEFAULT_NUM_EPOCHS
    end

    def l_rate
      @learning_rate ||= DEFAULT_LEARNING_RATE
    end

    def num_training_sets(offset = 0)
      NUM_TRAINING_SETS + offset
    end

    def training_inputs
      @data ||= [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    end

    def training_outputs
      @expected ||= [[0.0], [1.0], [1.0], [0.0]]
    end

    def training_set_initial_order
      Array.new(NUM_TRAINING_SETS) { |i| i }
    end

    # code..
    #
    def sigmoid(x)
      1.0 / (1.0 + exp(-x))
    end

    def dSigmoid(x)
      x * (1.0 - x)
    end

    def init_weight
      rand
    end

    def create_hidden_layer
      Layer.new(num_hidden_nodes, init_weight)
    end

    def create_output_layer
      Layer.new(num_outputs, init_weight)
    end

    def create_hidden_layer_bias
      Layer.new(num_hidden_nodes, init_weight)
    end

    def create_output_layer_bias
      Layer.new(num_outputs, init_weight)
    end

    def create_hidden_weights
      Weight.new(num_inputs, Weight.new(num_hidden_nodes, init_weight))
    end

    def create_output_weights
      Weight.new(num_hidden_nodes, Weight.new(num_outputs, init_weight))
    end

    def forward_inputs(data)
      @hidden_layer_bias.each_with_index do |hlb, hl_idx|
        activation = hlb
        data.each_with_index do |ti, t_idx|
          activation += ti * @hidden_weights[t_idx][hl_idx]
        end
        @hidden_layer[hl_idx] = sigmoid(activation)
      end
    end

    def forward_outputs
      @output_layer_bias.each_with_index do |lb, hl_idx|
        activation = lb
        @hidden_layer.each_with_index do |hl, t_idx|
          activation += hl * @output_weights[t_idx][hl_idx]
        end
        @output_layer[hl_idx] = sigmoid(activation)
      end
    end

    def forwards_prop(training_set)
      progress("Forward Inputs")
      forward_inputs(training_inputs[training_set])
      progress("Forward Outputs")
      forward_outputs
    end

    def backwards_outputs(training_set)
      delta_output = Array.new(num_outputs)
      delta_output.each_index do |idx|
        delta_output[idx] = (training_outputs[training_set][idx] - @output_layer[idx]) * dSigmoid(@output_layer[idx])
      end
    end

    def backwards_hidden(delta_output)
      delta_hidden = Array.new(num_hidden_nodes)
      delta_hidden.each_index do |h_idx|
        delta_hidden[h_idx] = 0.0
        delta_output.each_index do |o_idx|
          delta_hidden[h_idx] += delta_output[o_idx] * @output_weights[h_idx][o_idx]
        end
        delta_hidden[h_idx] = delta_hidden[h_idx] * dSigmoid(@hidden_layer[h_idx])
      end
    end

    def update_output(delta_output)
      delta_output.each_with_index do |d_o, idx|
        @output_layer_bias[idx] += delta_output[idx] * l_rate
        @hidden_layer.each_with_index do |hl, hl_idx|
          @output_weights[hl_idx][idx] += @hidden_layer[hl_idx] * d_o * l_rate
        end
      end
    end

    def update_hidden(delta_hidden, training_set)
      delta_hidden.each_index do |h_idx|
        @hidden_layer_bias[h_idx] += delta_hidden[h_idx] * l_rate
        (1..num_inputs).each do |i_idx|
          @hidden_weights[h_idx][i_idx - 1] += training_inputs[training_set][i_idx - 1] * delta_hidden[h_idx] * l_rate
        end
      end

    end

    def backwards_prop(training_set)
      progress("Backwards from Outputs")
      delta_output = backwards_outputs(training_set)
      progress("Backwards to Hidden Layers")
      delta_hidden = backwards_hidden(delta_output)

      update_output(delta_output)
      update_hidden(delta_hidden, training_set)

    end

    def progress(marker)
      return unless @debug
      puts "==========="
      puts marker
      puts "Hidden weights: #{@hidden_weights}"
      puts "Hidden biases: #{@hidden_layer_bias}"
      puts "Output weights: #{@output_weights}"
      puts "Output biases: #{@output_layer_bias}"
      puts "==========="
    end

  end
end
