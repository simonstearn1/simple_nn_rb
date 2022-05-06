require 'minitest/autorun'
require 'byebug'
require_relative '../fish.rb'

class FishTest < Minitest::Test
  def self.prepare
    puts "Moving all private methods to public for testing.."
    Fish::Net.send(:public, *Fish::Net.private_instance_methods)
  end
  prepare

  def test_integer_setup
    fish = Fish::Net.new
    [:num_inputs, :num_hidden_nodes, :num_outputs, :num_epochs, :num_training_sets].each do | setup_method |
      #puts setup_method
      assert fish.send(setup_method).is_a? Integer
    end
  end

  def test_float_setup
    fish = Fish::Net.new
    [:l_rate, :init_weight].each do | setup_method |
      #puts setup_method
      assert fish.send(setup_method).is_a? Float
    end
  end

  def test_array_setup
    fish = Fish::Net.new
    [:training_inputs, :training_outputs, :training_set_initial_order].each do | setup_method |
      #puts setup_method
      assert fish.send(setup_method).is_a? Array
    end
  end

  def test_training_set_initial_order
    fish = Fish::Net.new
    assert fish.training_set_initial_order.is_a? Array
    assert fish.training_set_initial_order.length == fish.num_training_sets
    fish.training_set_initial_order.each do | index |
      assert index.is_a? Integer
    end
  end

  def test_layer_setup
    fish = Fish::Net.new
    [:create_hidden_layer, :create_output_layer, :create_hidden_layer_bias, :create_output_layer_bias].each do | setup_method |
      #puts setup_method
      assert fish.send(setup_method).is_a? Fish::Layer
    end
  end

 def test_weights_setup
    fish = Fish::Net.new
    [:create_hidden_weights,  :create_output_weights].each do | setup_method |
      #puts setup_method
      assert fish.send(setup_method).is_a? Fish::Weight
    end
  end

end
