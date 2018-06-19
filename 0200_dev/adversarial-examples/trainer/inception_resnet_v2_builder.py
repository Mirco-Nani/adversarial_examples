import inception_resnet_v2_multilevel as base_model
import tensorflow as tf
from tensorflow.contrib import layers
slim = tf.contrib.slim

class InceptionResnetV2Builder:
    # Base enpoints are the names of the fully convolutional levels.
    # A network made only of these levels can process images of any size.
    BASE_ENDPOINTS = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','MaxPool_3a_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3','MaxPool_5a_3x3','Mixed_5b','block35_1','block35_2','block35_3','block35_4','block35_5','block35_6','block35_7','block35_8','block35_9','block35_10','Mixed_6a','block17_1','block17_2','block17_3','block17_4','block17_5','block17_6','block17_7','block17_8','block17_9','block17_10','block17_11','block17_12','block17_13','block17_14','block17_15','block17_16','block17_17','block17_18','block17_19','block17_20','PreAuxLogits','Mixed_7a','block8_1','block8_2','block8_3','block8_4','block8_5','block8_6','block8_7','block8_8','block8_9','Block8','Conv2d_7b_1x1']
    
    # Head endpoints are the names of the levels added after the base endpoints.
    # A network built with these levels assumes an input of shape [None, 299, 299, 3]
    HEAD_ENDPOINTS = ['PreLogitsFlatten', 'Logits', 'Final']
    
    LOGITS_ENDPOINT = 'Logits'
    DEFAULT_LAYER_SHAPES = dict([
        ('Conv2d_1a_3x3', [None, 149, 149, 32]),
        ('Conv2d_2a_3x3', [None, 147, 147, 32]),
        ('Conv2d_2b_3x3', [None, 147, 147, 64]),
        ('MaxPool_3a_3x3', [None, 73, 73, 64]),
        ('Conv2d_3b_1x1', [None, 73, 73, 80]),
        ('Conv2d_4a_3x3', [None, 71, 71, 192]),
        ('MaxPool_5a_3x3', [None, 35, 35, 192]),
        ('Mixed_5b', [None, 35, 35, 320]),
        ('block35_1', [None, 35, 35, 320]),
        ('block35_2', [None, 35, 35, 320]),
        ('block35_3', [None, 35, 35, 320]),
        ('block35_4', [None, 35, 35, 320]),
        ('block35_5', [None, 35, 35, 320]),
        ('block35_6', [None, 35, 35, 320]),
        ('block35_7', [None, 35, 35, 320]),
        ('block35_8', [None, 35, 35, 320]),
        ('block35_9', [None, 35, 35, 320]),
        ('block35_10', [None, 35, 35, 320]),
        ('Mixed_6a', [None, 17, 17, 1088]),
        ('block17_1', [None, 17, 17, 1088]),
        ('block17_2', [None, 17, 17, 1088]),
        ('block17_3', [None, 17, 17, 1088]),
        ('block17_4', [None, 17, 17, 1088]),
        ('block17_5', [None, 17, 17, 1088]),
        ('block17_6', [None, 17, 17, 1088]),
        ('block17_7', [None, 17, 17, 1088]),
        ('block17_8', [None, 17, 17, 1088]),
        ('block17_9', [None, 17, 17, 1088]),
        ('block17_10', [None, 17, 17, 1088]),
        ('block17_11', [None, 17, 17, 1088]),
        ('block17_12', [None, 17, 17, 1088]),
        ('block17_13', [None, 17, 17, 1088]),
        ('block17_14', [None, 17, 17, 1088]),
        ('block17_15', [None, 17, 17, 1088]),
        ('block17_16', [None, 17, 17, 1088]),
        ('block17_17', [None, 17, 17, 1088]),
        ('block17_18', [None, 17, 17, 1088]),
        ('block17_19', [None, 17, 17, 1088]),
        ('block17_20', [None, 17, 17, 1088]),
        ('PreAuxLogits', [None, 17, 17, 1088]),
        ('Mixed_7a', [None, 8, 8, 2080]),
        ('block8_1', [None, 8, 8, 2080]),
        ('block8_2', [None, 8, 8, 2080]),
        ('block8_3', [None, 8, 8, 2080]),
        ('block8_4', [None, 8, 8, 2080]),
        ('block8_5', [None, 8, 8, 2080]),
        ('block8_6', [None, 8, 8, 2080]),
        ('block8_7', [None, 8, 8, 2080]),
        ('block8_8', [None, 8, 8, 2080]),
        ('block8_9', [None, 8, 8, 2080]),
        ('Block8', [None, 8, 8, 2080]),
        ('Conv2d_7b_1x1', [None, 8, 8, 1536]),
        ('PreLogitsFlatten', [None, 1536])
    ])
    
    IMAGE_GRAPH_CHECKPOINT_URIS = [
        'gs://unipol-damage-machine-learning-retrain/unipol_retrain/nets/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt']
    
    CHECKPOINT_VARIABLES_TO_EXCLUDE_FROM_SCOPE = [
        'InceptionResnetV2/AuxLogits', 
        'InceptionResnetV2/Logits', 
        'global_step'
    ]
    
    def build_input(self, 
                    normalized=True,
                    # These constants are set by inception_resnet_v2 expectations
                    height = 299,
                    width = 299,
                    channels = 3,
                    batch=True):
        
        if batch:
            return self.build_batch_input(normalized=normalized, height=height, width=width, channels=channels)
        else:
            return self.build_single_input(normalized=normalized, height=height, width=width, channels=channels)
    
    def build_single_input(self, 
                    normalized=True,
                    # These constants are set by inception_resnet_v2 expectations
                    height = 299,
                    width = 299,
                    channels = 3):
        input_jpeg = tf.placeholder(tf.string, shape=None)
        image = tf.image.decode_jpeg(input_jpeg, channels=channels)

        # Note resize expects a batch_size, but we are feeding a single image.
        # So we have to expand then squeeze.  Resize returns float32 in the
        # range [0, uint8_max]
        image = tf.expand_dims(image, 0)

        # convert_image_dtype also scales [0, uint8_max] -> [0 ,1).
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(
            image, [height, width], align_corners=False)

        # Then rescale range to [-1, 1) for Inception.
        if normalized:
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            
        return input_jpeg, image
    
    def build_batch_input(self, 
                    normalized=True,
                    # These constants are set by inception_resnet_v2 expectations
                    height = 299,
                    width = 299,
                    channels = 3):
        
        image_str_tensor = tf.placeholder(tf.string, shape=[None])

        # The CloudML Prediction API always "feeds" the Tensorflow graph with
        # dynamic batch sizes e.g. (?,).  decode_jpeg only processes scalar
        # strings because it cannot guarantee a batch of images would have
        # the same output size.  We use tf.map_fn to give decode_jpeg a scalar
        # string from dynamic batches.
        def decode_and_resize(image_str_tensor):
            """Decodes jpeg string, resizes it and returns a uint8 tensor."""
            image = tf.image.decode_jpeg(image_str_tensor, channels=channels)
            # Note resize expects a batch_size, but tf_map supresses that index,
            # thus we have to expand then squeeze.  Resize returns float32 in the
            # range [0, uint8_max]
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, squeeze_dims=[0])
            image = tf.cast(image, dtype=tf.uint8)
            return image

        image = tf.map_fn(
            decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
        # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Then shift images to [-1, 1) for Inception.
        #if normalized:
        #    image = tf.subtract(image, 0.5)
        #    image = tf.multiply(image, 2.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image_str_tensor, image
    
    
    def build_train_model(self,
        inputs, 
        num_classes=0, 
        is_training=True,
        dropout_keep_prob=0.8,
        reuse=None, 
        scope='InceptionResnetV2',
        final_endpoint='Final', 
        final_layer_type="Softmax", #Sigmoid
        reverse=False
    ):
        level_name = final_endpoint
        
        net = inputs
        end_points = {}
        ordered_end_points = []
        
        layer_found = {"value" : final_endpoint in self.BASE_ENDPOINTS}
        def add_and_check_final(name, net):
            #global layer_found

            if reverse:
                #print(name,final_endpoint,name == final_endpoint, layer_found)
                if layer_found["value"]:
                    end_points[name] = net
                    ordered_end_points.append(name)

                if name == final_endpoint:
                    layer_found["value"] = True
                return False
            else:
                end_points[name] = net
                ordered_end_points.append(name)
                return name == final_endpoint


        def should_build():
            if reverse:
                return layer_found["value"]
            return True
        
        #with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],reuse=reuse) as scope:
        with slim.arg_scope(base_model.inception_resnet_v2_arg_scope()):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                should_build_base = not reverse or (reverse and final_endpoint in self.BASE_ENDPOINTS)
                if should_build_base:
                    net, end_points, ordered_end_points = base_model.inception_resnet_v2_base(net, 
                                                                                       final_endpoint=level_name, 
                                                                                       scope=scope,
                                                                                       reverse=reverse,
                                                                                       reuse=reuse)
                    should_stop_base = not reverse and final_endpoint in self.BASE_ENDPOINTS
                    if should_stop_base:
                        return net, end_points, ordered_end_points
                
                
                
                with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],reuse=reuse) as scope:
                    if should_build():
                        with tf.variable_scope('PreLogitsFlatten'):
                            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',scope='AvgPool_1a_8x8')
                            net = slim.flatten(net)

                            if is_training:
                                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='Dropout')

                            #end_points['PreLogitsFlatten'] = net
                    if add_and_check_final('PreLogitsFlatten', net): return net, end_points, ordered_end_points

                    if should_build():
                        if num_classes <= 0:
                            raise ValueError('num_classes must be specified to build level: %s', final_endpoint)

                        hidden_layer_size = 384 #1536/4

                        with tf.variable_scope('NewLogits'):
                            with tf.name_scope('Wx_plus_b'):
                                net = layers.fully_connected(net, hidden_layer_size)
                                # We need a dropout when the size of the dataset is rather small.
                                if dropout_keep_prob:
                                    #net = tf.nn.dropout(net, dropout_keep_prob)
                                    if is_training:
                                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training)
                                net = layers.fully_connected(net, num_classes, activation_fn=None)
                    if add_and_check_final('Logits', net): return net, end_points, ordered_end_points

                    if should_build():
                        print final_layer_type, 'Adding final layer'
                        if final_layer_type == 'Softmax':
                            net = tf.nn.softmax(net, name='Final')
                        elif final_layer_type == 'Sigmoid':
                            net = tf.nn.sigmoid(net, name='Final')
                        else:
                            raise ValueError("Softmax or Sigmoid!!!!!!")
                    if add_and_check_final('Final', net): return net, end_points, ordered_end_points
                
        if not reverse:
            raise ValueError('final_endpoint (%s) not recognized', final_endpoint)
        else:
            return net, end_points, ordered_end_points
        
        
        
    def build_predict_model(self,
        inputs, 
        num_classes=0, 
        #is_training=True,
        #dropout_keep_prob=0.8,
        reuse=None, 
        scope='InceptionResnetV2',
        final_endpoint='Final',
        final_layer_type="Softmax",
        reverse=False
    ):
        level_name = final_endpoint
        
        net = inputs
        end_points = {}
        ordered_end_points = []
        
        layer_found = {"value" : final_endpoint in self.BASE_ENDPOINTS}
        def add_and_check_final(name, net):
            #global layer_found

            if reverse:
                #print(name,final_endpoint,name == final_endpoint, layer_found)
                if layer_found["value"]:
                    end_points[name] = net
                    ordered_end_points.append(name)

                if name == final_endpoint:
                    layer_found["value"] = True
                return False
            else:
                end_points[name] = net
                ordered_end_points.append(name)
                return name == final_endpoint


        def should_build():
            if reverse:
                return layer_found["value"]
            return True
        
        #with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],reuse=reuse) as scope:
        with slim.arg_scope(base_model.inception_resnet_v2_arg_scope()):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=False):
                should_build_base = not reverse or (reverse and final_endpoint in self.BASE_ENDPOINTS)
                if should_build_base:
                    net, end_points, ordered_end_points = base_model.inception_resnet_v2_base(net, 
                                                                                       final_endpoint=level_name, 
                                                                                       scope=scope,
                                                                                       reverse=reverse,
                                                                                       reuse=reuse)
                    should_stop_base = not reverse and final_endpoint in self.BASE_ENDPOINTS
                    if should_stop_base:
                        return net, end_points, ordered_end_points
                
                
                
                with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],reuse=reuse) as scope:
                    if should_build():
                        with tf.variable_scope('PreLogitsFlatten'):
                            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',scope='AvgPool_1a_8x8')
                            net = slim.flatten(net)

                            #if is_training:
                            #    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='Dropout')

                            #end_points['PreLogitsFlatten'] = net
                    if add_and_check_final('PreLogitsFlatten', net): return net, end_points, ordered_end_points

                    if should_build():
                        if num_classes <= 0:
                            raise ValueError('num_classes must be specified to build level: %s', final_endpoint)

                        hidden_layer_size = 384 #1536/4

                        with tf.variable_scope('NewLogits'):
                            with tf.name_scope('Wx_plus_b'):
                                net = layers.fully_connected(net, hidden_layer_size)
                                # We need a dropout when the size of the dataset is rather small.
                                #if dropout_keep_prob:
                                #    #net = tf.nn.dropout(net, dropout_keep_prob)
                                #    if is_training:
                                #        net = slim.dropout(net, dropout_keep_prob, is_training=is_training)
                                net = layers.fully_connected(net, num_classes, activation_fn=None)
                    if add_and_check_final('Logits', net): return net, end_points, ordered_end_points

                    if should_build():
                        #print final_layer_type, 'Adding final layer'
                        if final_layer_type == 'Softmax':
                            net = tf.nn.softmax(net, name='Final')
                        elif final_layer_type == 'Sigmoid':
                            net = tf.nn.sigmoid(net, name='Final')
                        else:
                            raise ValueError("Softmax or Sigmoid!!!!!!")
                    if add_and_check_final('Final', net): return net, end_points, ordered_end_points
                
        if not reverse:
            raise ValueError('final_endpoint (%s) not recognized', final_endpoint)
        else:
            return net, end_points, ordered_end_points