use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 打印所有可用的张量名称以便调试
        println!("Available tensors: {:?}", safetensor.names());
        
        // 辅助函数：从bytes转换为f32数组
        let bytes_to_f32 = |bytes: &[u8]| -> Vec<f32> {
            bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };

        // 辅助函数：加载张量，添加错误处理
        let get_tensor = |name: &str, shape: &[usize]| -> Tensor<f32> {
            match safetensor.tensor(name) {
                Ok(tensor_view) => {
                    Tensor::new(bytes_to_f32(tensor_view.data()), &shape.to_vec())
                },
                Err(_) => {
                    // 尝试不同的命名格式
                    let alt_name = if name.starts_with("model.") {
                        name.strip_prefix("model.").unwrap()
                    } else {
                        name
                    };
                    match safetensor.tensor(alt_name) {
                        Ok(tensor_view) => Tensor::new(bytes_to_f32(tensor_view.data()), &shape.to_vec()),
                        Err(_) => panic!("Tensor not found: {} or {}", name, alt_name)
                    }
                }
            }
        };

        // 基础参数
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let intermediate_size = config.intermediate_size;

        // 使用 lm_head.weight 作为embedding表
        let embedding_table = get_tensor(
            "lm_head.weight",
            &[vocab_size, hidden_size]
        );
        let lm_head = get_tensor(
            "lm_head.weight",
            &[vocab_size, hidden_size]
        );

        // 加载输出层norm权重
        let rms_out_w = get_tensor(
            "model.norm.weight",
            &[hidden_size]
        );

        // 初始化层参数向量
        let mut rms_att_w = Vec::with_capacity(num_layers);
        let mut rms_ffn_w = Vec::with_capacity(num_layers);
        let mut wq = Vec::with_capacity(num_layers);
        let mut wk = Vec::with_capacity(num_layers);
        let mut wv = Vec::with_capacity(num_layers);
        let mut wo = Vec::with_capacity(num_layers);
        let mut w_gate = Vec::with_capacity(num_layers);
        let mut w_up = Vec::with_capacity(num_layers);
        let mut w_down = Vec::with_capacity(num_layers);

        // 按层加载参数，保持原有前缀
        for i in 0..num_layers {
            let prefix = format!("model.layers.{}", i);
            
            // Attention相关参数
            rms_att_w.push(get_tensor(
                &format!("{}.input_layernorm.weight", prefix),
                &[hidden_size]
            ));
            wq.push(get_tensor(
                &format!("{}.self_attn.q_proj.weight", prefix),
                &[hidden_size, hidden_size]
            ));
            wk.push(get_tensor(
                &format!("{}.self_attn.k_proj.weight", prefix),
                &[hidden_size, hidden_size]
            ));
            wv.push(get_tensor(
                &format!("{}.self_attn.v_proj.weight", prefix),
                &[hidden_size, hidden_size]
            ));
            wo.push(get_tensor(
                &format!("{}.self_attn.o_proj.weight", prefix),
                &[hidden_size, hidden_size]
            ));

            // FFN相关参数
            rms_ffn_w.push(get_tensor(
                &format!("{}.post_attention_layernorm.weight", prefix),
                &[hidden_size]
            ));
            w_gate.push(get_tensor(
                &format!("{}.mlp.gate_proj.weight", prefix),
                &[intermediate_size, hidden_size]
            ));
            w_up.push(get_tensor(
                &format!("{}.mlp.up_proj.weight", prefix),
                &[intermediate_size, hidden_size]
            ));
            w_down.push(get_tensor(
                &format!("{}.mlp.down_proj.weight", prefix),
                &[hidden_size, intermediate_size]
            ));
        }

        Self {
            embedding_table,
            lm_head,
            rms_att_w,
            rms_ffn_w,
            rms_out_w,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
        }
    }
}
