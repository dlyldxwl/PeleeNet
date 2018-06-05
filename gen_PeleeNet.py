#coding: utf-8
#by Chen yh

tran_channel = 32  #growth_rate
class Genpelee():

    def __init__(self):
        self.last = "data"

    def header(self,name):
        s_name = "name: \"%s\"" %name
        return s_name

    def input(self,size):
        s_input = """
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: %d
      dim: %d
    }
  }
}""" %(size,size)
        return s_input

    def conv(self, name, num_output, k_size, stride, bias_term = False, bottom = None):

        if bottom is None:
            bottom = self.last

        if bias_term:
            bias_str = ""
            bias_filler = """
    bias_filler {
      type: "constant"
    }        
"""
        else:
            bias_str = "\n    bias_term: false"
            bias_filler = ""

        if k_size>1 :
            pad_str = "\n    pad: %d" %(int(k_size/2))
        else:
            pad_str = ""

        s_conv = """
layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  convolution_param {
    num_output: %d%s%s
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "xavier"
    }%s
  }
}""" %(name,bottom,name,num_output,bias_str,pad_str,k_size,stride,bias_filler)

        self.last = name
        return s_conv

    def bn(self,bottom = None):

        if bottom is None:
            bottom = self.last

        s_bn = """
layer {
  name: "%s/bn"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 0.0
    decay_mult: 0.0
  }
  param {
    lr_mult: 0.0
    decay_mult: 0.0
  }
  param {
    lr_mult: 0.0
    decay_mult: 0.0
  }
  batch_norm_param {
    moving_average_fraction: 0.999000012875
    eps: 0.0010000000475
  }
}
layer {
  name: "%s/scale"
  type: "Scale"
  bottom: "%s"
  top: "%s"
  scale_param {
    filler {
      value: 1.0
    }
    bias_term: true
    bias_filler {
      value: 0.0
    }
  }
}""" %(bottom,bottom,bottom,bottom,bottom,bottom)
        return s_bn

    def relu(self,bottom = None):

        if bottom is None:
            bottom = self.last

        s_relu = """
layer {
  name: "%s/relu"
  type: "ReLU"
  bottom: "%s"
  top: "%s"
}""" %(bottom,bottom,bottom)
        return s_relu

    def pool(self, name, k_size, stride, bottom = None):
        if bottom is None:
            bottom = self.last

        s_pool = """
layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: MAX
    kernel_size: %d
    stride: %d
  }
}""" %(name,bottom,name,k_size,stride)
        self.last = name
        return s_pool

    def ave_pool(self, bottom = None):
        if bottom is None:
            bottom = self.last

        s_avepool = """
layer {
  name: "global_pool"
  type: "Pooling"
  bottom: "%s"
  top: "global_pool"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}""" %(bottom)
        self.last = "global_pool"
        return s_avepool

    def cls(self, num = 1000, bottom = "global_pool"):
        s_cls = """
layer {
  name: "classifier"
  type: "InnerProduct"
  bottom: "%s"
  top: "classifier"
  inner_product_param {
    num_output: %d
    bias_term: true
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "classifier"
  top: "prob"
}""" %(bottom,num)
        return s_cls

    def concat(self,bottom,top):
        assert type(bottom) is list

        length = len(bottom)
        str_bo = ""
        for i in range(length):
            str_bo += """
  bottom: "%s" """ %(bottom[i])

        s_concat = """
layer {
  name: "%s"
  type: "Concat"%s
  top: "%s"
  concat_param {
    axis: 1
  }
}""" %(top,str_bo,top)
        self.last = top
        return s_concat

    def conv_block(self, name, num_output, k_size, stride, bottom = None):
        if bottom is None:
            bottom = self.last

        s_conv_block = self.conv(name, num_output, k_size, stride, bottom = bottom)+self.bn()+self.relu()
        return s_conv_block

    def stem_block(self,init_feature):

        s1 = self.conv_block("stem1",init_feature,3,2)
        s2_a = self.conv_block("stem2a",int(init_feature/2),1,1)
        s2_b = self.conv_block("stem2b",init_feature,3,2)
        s2_c = self.pool("stem1/pool",2,2,bottom= "stem1")
        s3 = self.concat(["stem2b","stem1/pool"],"stem/concat")+self.conv_block("stem",init_feature,1,1)
        stem = s1+s2_a+s2_b+s2_c+s3
        return stem

    def dense_block(self, num_layers, bottleneck_width, stage, growth_rate=32, bottom=None):
        if bottom is None:
            bottom = self.last

        g = int(growth_rate / 2)
        btn_width = g * bottleneck_width
        s_dense = ""

        for i in range(1, num_layers+1):
            base_name = "stage"+str(stage)+"_"+str(i)
            branch_1 = self.conv_block(base_name+"/a1",btn_width,1,1)+self.conv_block(base_name+"/a2",g,3,1)
            branch_2 = self.conv_block(base_name+"/b1",btn_width,1,1,bottom=bottom)+self.conv_block(base_name+"/b2",g,3,1)+self.conv_block(base_name+"/b3",g,3,1)
            concat = self.concat([bottom,base_name+"/a2",base_name+"/b3"],base_name)
            s_dense += branch_1 + branch_2 + concat
            bottom = self.last

        assert self.last == base_name
        return s_dense

    def transition_layer(self, growth_rate=32, has_pool = True, bottom = None):
        if bottom is None:
            bottom = self.last

        name,num = bottom.split("_",1)
        global tran_channel
        tran_channel += (int(num))*growth_rate
        s_tran = self.conv_block(name, tran_channel,1,1)

        if has_pool:
            s_tran += self.pool(name+"/pool",2,2)

        return s_tran

    def generate(self):
        s_steam = self.header("PeleeNet")+self.input(224)+self.stem_block(32)
        s_stage1 = self.dense_block(3,1,1)+self.transition_layer()
        s_stage2 = self.dense_block(4,2,2)+self.transition_layer()
        s_stage3 = self.dense_block(8,4,3)+self.transition_layer()
        s_stage4 = self.dense_block(6,4,4)+self.transition_layer(has_pool=False)
        s_cls = self.ave_pool()+self.cls()
        s_net = s_steam+s_stage1+s_stage2+s_stage3+s_stage4+s_cls
        return s_net

if __name__ == "__main__":
    gen = Genpelee()
    s= gen.generate()
    with open("PeleeNet.prototxt",'w') as a:
        a.write(s)
