client<llm> MlxCommunityQwen35_08bMlxBf16 {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8091/v1"
    model "mlx-community/Qwen3.5-0.8B-MLX-bf16"
    temperature 0
    max_tokens 1024
    chat_template_kwargs {
      enable_thinking false
    }
  }
}

client<llm> MlxCommunityQwen35_35bA3b8bit {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8092/v1"
    model "mlx-community/Qwen3.5-35B-A3B-8bit"
    temperature 0
    max_tokens 2048
    chat_template_kwargs {
      enable_thinking false
    }
  }
}

client<llm> LiquidAiLfm25_8bA1bMlx8bit {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8093/v1"
    model "LiquidAI/LFM2.5-8B-A1B-MLX-8bit"
    temperature 0
    max_tokens 2048
  }
}

client<llm> Gpt55Medium {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8080/v1"
    model "gpt-5.5"
    temperature 0
    max_tokens 2048
    reasoning_effort "medium"
  }
}

client<llm> Qwen35_08bSqlAgentVllm {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8094/v1"
    model "qwen3_5_0_8b_sql_agent"
    temperature 0
    max_tokens 1024
    chat_template_kwargs {
      enable_thinking false
    }
  }
}

client<llm> MlxCommunityQwen35_08bMlxBf16Lora {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8095/v1"
    model "mlx-community/Qwen3.5-0.8B-MLX-bf16"
    temperature 0
    max_tokens 1024
    chat_template_kwargs {
      enable_thinking false
    }
  }
}
