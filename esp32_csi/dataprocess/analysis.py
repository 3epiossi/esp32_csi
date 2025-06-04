"""
TensorFlow Lite Operations Analysis for Keras LSTM Model
分析哪些操作是必要的，哪些是多餘的

基於 Keras 模型結構:
- Reshape layer
- LSTM layer (16 units)
- Dense layer (32 units, ReLU)
- Concatenate layer
- Dense layer (output)
- Softmax layer
"""

class TFLiteOperationsAnalysis:
    def __init__(self):
        self.keras_operations = {
            'reshape': 'layers.Reshape((input_shape[0]//3, 3))',
            'lstm': 'layers.LSTM(16, return_sequences=False, activation="tanh")',
            'dense1': 'layers.Dense(32, activation="relu")',
            'concatenate': 'layers.Concatenate()',
            'dense2': 'layers.Dense(num_classes)',
            'softmax': 'layers.Softmax()'
        }
        
        self.registered_operations = [
            'AddStridedSlice',
            'AddFill', 
            'AddTanh',
            'AddWhile',
            'AddSlice',
            'AddMaximum',
            'AddSoftmax',
            'AddUnidirectionalSequenceLSTM',
            'AddPack',
            'AddGather',
            'AddLess',
            'AddTranspose',
            'AddShape',
            'AddFullyConnected',
            'AddAdd',
            'AddReshape',
            'AddSplit',
            'AddRelu',
            'AddConcatenation',
            'AddMul',
            'AddMinimum'
        ]

    def analyze_operation_necessity(self):
        """分析每個註冊操作的必要性"""
        
        analysis = {
            # 直接對應的必要操作
            'ESSENTIAL': {
                'AddReshape': {
                    'keras_equivalent': 'layers.Reshape()',
                    'reason': '直接對應 Reshape 層',
                    'usage': '將 1D 輸入轉換為 2D 序列'
                },
                'AddUnidirectionalSequenceLSTM': {
                    'keras_equivalent': 'layers.LSTM()',
                    'reason': '直接對應 LSTM 層',
                    'usage': '序列處理的核心操作'
                },
                'AddTanh': {
                    'keras_equivalent': 'activation="tanh" in LSTM',
                    'reason': 'LSTM 內部激活函數',
                    'usage': 'LSTM 的預設激活函數'
                },
                'AddFullyConnected': {
                    'keras_equivalent': 'layers.Dense()',
                    'reason': '對應 Dense 層',
                    'usage': '全連接層實現'
                },
                'AddRelu': {
                    'keras_equivalent': 'activation="relu"',
                    'reason': '對應 Dense 層的 ReLU 激活',
                    'usage': 'Dense1 層的激活函數'
                },
                'AddConcatenation': {
                    'keras_equivalent': 'layers.Concatenate()',
                    'reason': '直接對應 Concatenate 層',
                    'usage': '連接 LSTM 輸出和 Dense1 輸出'
                },
                'AddSoftmax': {
                    'keras_equivalent': 'layers.Softmax()',
                    'reason': '直接對應 Softmax 層',
                    'usage': '最終輸出的機率分布'
                }
            },
            
            # LSTM 內部實現需要的操作
            'LSTM_INTERNAL': {
                'AddAdd': {
                    'keras_equivalent': 'LSTM internal calculations',
                    'reason': 'LSTM 門控機制中的加法運算',
                    'usage': '遺忘門、輸入門、輸出門的計算'
                },
                'AddMul': {
                    'keras_equivalent': 'LSTM internal calculations', 
                    'reason': 'LSTM 門控機制中的乘法運算',
                    'usage': '門控值與狀態的逐元素乘法'
                },
                'AddWhile': {
                    'keras_equivalent': 'LSTM time step iteration',
                    'reason': 'LSTM 時間步迴圈控制',
                    'usage': '序列處理的迴圈結構'
                },
                'AddSlice': {
                    'keras_equivalent': 'LSTM internal tensor operations',
                    'reason': 'LSTM 內部張量切片操作',
                    'usage': '提取不同門控的權重和狀態'
                },
                'AddSplit': {
                    'keras_equivalent': 'LSTM gate separation',
                    'reason': 'LSTM 門控分離',
                    'usage': '將合併的門控輸出分離成不同門'
                },
                'AddTranspose': {
                    'keras_equivalent': 'LSTM matrix operations',
                    'reason': 'LSTM 權重矩陣轉置',
                    'usage': '矩陣乘法中的維度調整'
                }
            },
            
            # 輔助和優化操作
            'AUXILIARY': {
                'AddShape': {
                    'keras_equivalent': 'Dynamic shape inference',
                    'reason': '動態形狀推斷',
                    'usage': '運行時獲取張量維度信息'
                },
                'AddPack': {
                    'keras_equivalent': 'Tensor stacking',
                    'reason': '張量堆疊操作',
                    'usage': '可能用於批次處理或序列重組'
                },
                'AddGather': {
                    'keras_equivalent': 'Index-based tensor extraction',
                    'reason': '基於索引的張量提取',
                    'usage': '可能用於序列中的特定時間步提取'
                },
                'AddStridedSlice': {
                    'keras_equivalent': 'Advanced tensor slicing',
                    'reason': '進階張量切片',
                    'usage': '更複雜的張量子集提取'
                }
            },
            
            # 可能多餘的操作
            'POTENTIALLY_REDUNDANT': {
                'AddFill': {
                    'keras_equivalent': 'Tensor initialization',
                    'reason': '張量填充/初始化',
                    'usage': '初始化零張量或常數張量',
                    'redundancy_reason': '簡單模型可能不需要動態填充'
                },
                'AddMaximum': {
                    'keras_equivalent': 'Element-wise maximum',
                    'reason': '逐元素最大值運算',
                    'usage': '可能用於 ReLU 的實現 (max(0, x))',
                    'redundancy_reason': 'AddRelu 已經涵蓋了 ReLU 功能'
                },
                'AddMinimum': {
                    'keras_equivalent': 'Element-wise minimum',
                    'reason': '逐元素最小值運算',
                    'usage': '可能用於梯度裁剪或數值穩定性',
                    'redundancy_reason': '簡單推理模型通常不需要'
                },
                'AddLess': {
                    'keras_equivalent': 'Comparison operation',
                    'reason': '比較運算',
                    'usage': '可能用於條件判斷或迴圈控制',
                    'redundancy_reason': '簡單前饋可能不需要複雜條件'
                }
            }
        }
        
        return analysis

    def generate_minimal_resolver(self):
        """生成最小化的操作解析器"""
        
        minimal_ops = [
            # 核心必要操作
            'resolver.AddReshape();',
            'resolver.AddUnidirectionalSequenceLSTM();', 
            'resolver.AddTanh();',
            'resolver.AddFullyConnected();',
            'resolver.AddRelu();',
            'resolver.AddConcatenation();',
            'resolver.AddSoftmax();',
            
            # LSTM 內部必要操作
            'resolver.AddAdd();',
            'resolver.AddMul();',
            'resolver.AddWhile();',
            'resolver.AddSlice();',
            'resolver.AddSplit();',
            'resolver.AddTranspose();',
            
            # 輔助操作（謹慎移除）
            'resolver.AddShape();',
            'resolver.AddPack();',
            'resolver.AddGather();',
            'resolver.AddStridedSlice();'
        ]
        
        return minimal_ops

    def generate_redundancy_report(self):
        """生成多餘操作報告"""
        
        report = """
=== TensorFlow Lite 操作多餘性分析報告 ===

基於 Keras 模型結構分析，以下是操作的分類：

✅ 必要操作 (ESSENTIAL):
- AddReshape: 對應 Reshape 層
- AddUnidirectionalSequenceLSTM: 對應 LSTM 層
- AddTanh: LSTM 激活函數
- AddFullyConnected: 對應 Dense 層
- AddRelu: Dense 層激活函數
- AddConcatenation: 對應 Concatenate 層
- AddSoftmax: 對應 Softmax 層

🔧 LSTM 內部操作 (LSTM_INTERNAL):
- AddAdd: LSTM 門控計算中的加法
- AddMul: LSTM 門控計算中的乘法
- AddWhile: LSTM 時間步迴圈
- AddSlice: LSTM 內部張量切片
- AddSplit: LSTM 門控分離
- AddTranspose: LSTM 矩陣運算

⚙️ 輔助操作 (AUXILIARY):
- AddShape: 動態形狀推斷
- AddPack: 張量堆疊
- AddGather: 索引提取
- AddStridedSlice: 進階切片

❓ 可能多餘 (POTENTIALLY_REDUNDANT):
- AddFill: 張量填充（簡單模型可能不需要）
- AddMaximum: 逐元素最大值（AddRelu 已涵蓋）
- AddMinimum: 逐元素最小值（通常不需要）
- AddLess: 比較運算（簡單前饋不需要）

建議移除的操作:
1. AddFill - 除非有動態張量初始化需求
2. AddMaximum - AddRelu 已經實現 ReLU 功能
3. AddMinimum - 一般推理不需要
4. AddLess - 簡單模型不需要條件判斷

注意事項:
- LSTM 內部實現複雜，建議保留所有 LSTM 相關操作
- 輔助操作雖然看似非必要，但可能在運行時動態使用
- 建議從完整操作集開始，逐步測試移除的可行性
        """
        
        return report

    def estimate_memory_savings(self):
        """估算移除多餘操作後的記憶體節省"""
        
        # 每個操作大約的記憶體開銷（估算值）
        op_memory_estimate = {
            'AddFill': 512,        # bytes
            'AddMaximum': 256,     # bytes  
            'AddMinimum': 256,     # bytes
            'AddLess': 128,        # bytes
        }
        
        total_savings = sum(op_memory_estimate.values())
        
        savings_report = f"""
記憶體節省估算:
- AddFill: ~{op_memory_estimate['AddFill']} bytes
- AddMaximum: ~{op_memory_estimate['AddMaximum']} bytes
- AddMinimum: ~{op_memory_estimate['AddMinimum']} bytes  
- AddLess: ~{op_memory_estimate['AddLess']} bytes

總計節省: ~{total_savings} bytes ({total_savings/1024:.1f} KB)

注意: 這是粗略估算，實際節省可能因實現而異
        """
        
        return savings_report

# 使用範例
if __name__ == "__main__":
    analyzer = TFLiteOperationsAnalysis()
    
    print("=== 操作必要性分析 ===")
    analysis = analyzer.analyze_operation_necessity()
    
    for category, ops in analysis.items():
        print(f"\n{category}:")
        for op_name, details in ops.items():
            print(f"  {op_name}:")
            print(f"    - 對應: {details['keras_equivalent']}")
            print(f"    - 原因: {details['reason']}")
            if 'redundancy_reason' in details:
                print(f"    - 多餘原因: {details['redundancy_reason']}")
    
    print("\n" + "="*50)
    print("最小化操作建議:")
    minimal_ops = analyzer.generate_minimal_resolver()
    for op in minimal_ops:
        print(f"  {op}")
    
    print("\n" + "="*50)
    print(analyzer.generate_redundancy_report())
    
    print("\n" + "="*50)
    print(analyzer.estimate_memory_savings())