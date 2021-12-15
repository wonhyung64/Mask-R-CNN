#%%
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Dense, Flatten, Dropout, BatchNormalization, Conv2DTranspose, Activation
#%%
class RPN(Model):
    
    def __init__(self, hyper_params):
        super(RPN, self).__init__()
        self.hyper_params = hyper_params

        self.base_model = VGG16(include_top=False, input_shape=(self.hyper_params["img_size"], 
                                                                self.hyper_params["img_size"],
                                                                3))        

        self.layer = self.base_model.get_layer('block5_conv3').output

        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False

        self.conv = Conv2D(filters=512, kernel_size=(3, 3), 
                           activation='relu', padding='same', 
                           name='rpn_conv')

        self.rpn_cls_output = Conv2D(filters=self.hyper_params['anchor_count'], 
                                     kernel_size=(1, 1), 
                                     activation='sigmoid', 
                                     name='rpn_cls')

        self.rpn_reg_output = Conv2D(filters=self.hyper_params['anchor_count']*4, 
                                     kernel_size=(1,1), 
                                     activation='linear', 
                                     name='rpn_reg')

    def call(self,inputs):
        feature_map = self.feature_extractor(inputs) 
        x = self.conv(feature_map)
        cls = self.rpn_cls_output(x)
        reg = self.rpn_reg_output(x)
        return [reg, cls, feature_map]
#%%
class DTN(Model):
    def __init__(self, hyper_params):
        super(DTN, self).__init__()
        self.hyper_params = hyper_params
        #
        self.br1_FC1 = TimeDistributed(Flatten(), name='frcnn_flatten')
        self.br1_FC2 = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc1')
        self.br1_FC3 = TimeDistributed(Dropout(0.5), name='frcnn_dropout1')
        self.br1_FC4 = TimeDistributed(Dense(4096, activation='relu'), name='frcnn_fc2')
        self.br1_FC5 = TimeDistributed(Dropout(0.5), name='frcnn_dropout2')
        #
        self.br1_cls = TimeDistributed(Dense(self.hyper_params['total_labels'], 
                                         activation='softmax'), 
                                         name='frcnn_cls')
        self.br1_reg = TimeDistributed(Dense(self.hyper_params['total_labels'] * 4, 
                                         activation='linear'), 
                                         name='frcnn_reg')

        self.hyper_params = hyper_params
        self.br2_cv1 = TimeDistributed(Conv2D(256, (3,3), padding="same", name="mrcnn_mask_conv1"))
        self.br2_bn1 = TimeDistributed(BatchNormalization(),name = 'mrcnn_mask_bn1')
        self.br2_ac1 = Activation("relu")
        self.br2_cv2 = TimeDistributed(Conv2D(256, (3,3), padding="same"), name="mrcnn_mask_conv2")
        self.br2_bn2 = TimeDistributed(BatchNormalization(), name="mrcnn_mask_bn2")
        self.br2_ac2 = Activation("relu")
        self.br2_cv3 = TimeDistributed(Conv2D(256, (3,3), padding="same"), name="mrcnn_mask_conv3")
        self.br2_bn3 = TimeDistributed(BatchNormalization(),name = 'mrcnn_mask_bn3')
        self.br2_ac3 = Activation("relu")
        self.br2_cv4 = TimeDistributed(Conv2D(256, (3,3), padding="same"), name="mrcnn_mask_conv4")
        self.br2_bn4 = TimeDistributed(BatchNormalization(), name="mrcnn_mask_bn4")
        self.br2_ac4 = Activation("relu")
        self.br2_dcv = TimeDistributed(Conv2DTranspose(256, (2,2), strides=2, activation="relu"), name="mrcnn_mask_deconv")
        self.br2_msk = TimeDistributed(Conv2D(self.hyper_params["total_labels"], (1,1), strides=1, activation="sigmoid"), name="mrcnn_mask")

    def call(self, inputs):
        br1_fc1 = self.br1_FC1(inputs)
        br1_fc2 = self.br1_FC2(br1_fc1)
        br1_fc3 = self.br1_FC3(br1_fc2)
        br1_fc4 = self.br1_FC4(br1_fc3)
        br1_fc5 = self.br1_FC5(br1_fc4)
        cls = self.br1_cls(br1_fc5)
        reg = self.br1_reg(br1_fc5)

        br2_cv1 = self.br2_cv1(inputs)
        br2_bn1 = self.br2_bn1(br2_cv1)
        br2_ac1 = self.br2_ac1(br2_bn1)
        br2_cv2 = self.br2_cv2(br2_ac1)
        br2_bn2 = self.br2_bn2(br2_cv2)
        br2_ac2 = self.br2_ac2(br2_bn2)
        br2_cv3 = self.br2_cv3(br2_ac2)
        br2_bn3 = self.br2_bn3(br2_cv3)
        br2_ac3 = self.br2_ac3(br2_bn3)
        br2_cv4 = self.br2_cv4(br2_ac3)
        br2_bn4 = self.br2_bn4(br2_cv4)
        br2_ac4 = self.br2_ac4(br2_bn4)
        br2_dcv = self.br2_dcv(br2_ac4)
        msk = self.br2_msk(br2_dcv)

        return [reg, cls, msk]