module Fish

  class FishException < StandardError

  end

  class NetworkNotTrained < FishException
    def initialize(msg="This network has not yet been trained (train method).", exception_type="out_of_order")
      @exception_type = exception_type
      super(msg)
    end
  end

  class SampleNotArray < FishException
    def initialize(msg="The supplied sample is not an array.", exception_type="data_type")
      @exception_type = exception_type
      super(msg)
    end
  end

  class SampleWrongLength < FishException
    def initialize(msg="The supplied sample has the wrong number of items (mismatch with expected inputs).", exception_type="data_size")
      @exception_type = exception_type
      super(msg)
    end
  end

  class SampleContainsNilValues < FishException
    def initialize(msg="The supplied sample has nil values when inputs are expected (if you are missing values try guessing with mean/median/mode/best guess).", exception_type="unexpected_nil")
      @exception_type = exception_type
      super(msg)
    end
  end

  class UnknownActivationFunction < FishException
    def initialize(msg="Supported activation functions are :sigmoid, :tanh and :relu.", exception_type="activation_invalid")
      @exception_type = exception_type
      super(msg)
    end
  end

end
