# 匯入 fast.ai 和 transformers 的套件
from fastai.text.all import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 定義一個函數，用來將文本資料轉換成 T5 模型可以接受的格式
def t5_preprocess(text):
    # 在文本前面加上 "summarize: " 的字串，讓模型知道要做摘要的任務
    text = "summarize: " + text
    # 使用 T5 的分詞器來將文本分割成一個一個的詞彙
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokens = tokenizer.tokenize(text)
    # 將分詞後的詞彙轉換成數字，並限制最大長度為 512
    ids = tokenizer.convert_tokens_to_ids(tokens)[:512]
    # 將數字轉換成張量 (tensor) 的格式，並加上一個維度，方便之後的批次處理
    input_ids = torch.LongTensor(ids).unsqueeze(0)
    # 回傳轉換後的張量
    return input_ids

# 定義一個函數，用來將模型的輸出轉換成文本格式
def t5_postprocess(output_ids):
    # 使用 T5 的分詞器來將數字轉換成詞彙
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokens = tokenizer.convert_ids_to_tokens(output_ids)
    # 將詞彙連接成一個字串，並移除不必要的符號
    text = tokenizer.convert_tokens_to_string(tokens)
    text = text.replace("<pad>", "").replace("</s>", "").strip()
    # 回傳轉換後的字串
    return text

# 建立一個 T5 模型的物件，並載入預訓練的權重
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 建立一個資料集的物件，這裡使用 fast.ai 內建的一個文本分類資料集，只取其中的文本部分
# 這個資料集包含了一些電影評論的文本，我們的目標是從這些文本中生成摘要或重點
dset = TextDataLoaders.from_folder(path="C:\Users\User\Desktop\AI\data.txt", text_vocab=Vocab.create())
dset = dset.train_ds.map(lambda x: x[0])

# 使用 DataLoader 類別來建立一個資料載入器的物件，這個物件可以幫助我們將資料分成一個一個的批次
# 我們指定了以下的參數：
# dataset: 要使用的資料集，這裡是我們剛剛建立的 dset
# bs: 每個批次的資料數量，這裡是 8，也就是每次訓練或生成時會使用 8 個文本
# shuffle: 是否要隨機打亂資料的順序，這裡是 True，這樣可以增加訓練的隨機性和多樣性
# collate_fn: 用來將資料轉換成模型可以接受的格式的函數，這裡是我們自己定義的 t5_preprocess
dl = DataLoader(dataset=dset, bs=8, shuffle=True, collate_fn=t5_preprocess)

# 定義一個函數，用來訓練模型
def train_model(model, dl, epochs):
    # 將模型切換到訓練模式
    model.train()
    # 遍歷指定的訓練回合數
    for epoch in range(epochs):
        # 初始化一個累積的損失值
        total_loss = 0
        # 遍歷資料載入器中的每一個批次
        for input_ids in dl:
            # 將輸入的張量放到 GPU 上，如果有的話
            input_ids = input_ids.to("cuda") if torch.cuda.is_available() else input_ids
            # 使用模型的 forward 方法來計算輸出和損失值
            # 我們指定了以下的參數：
            # input_ids: 輸入的張量，包含了文本的數字表示
            # labels: 標籤的張量，也就是我們想要生成的摘要或重點，這裡我們直接使用輸入的張量作為標籤，讓模型嘗試重建輸入的文本
            # reduction: 計算損失值時的縮減方式，這裡是 "mean"，也就是對所有的損失值取平均
            output = model(input_ids=input_ids, labels=input_ids, reduction="mean")
            # 取出損失值
            loss = output.loss
            # 將損失值累加到總損失值中
            total_loss += loss.item()
            # 清空梯度值
            model.zero_grad()
            # 反向傳播計算梯度值
            loss.backward()
            # 使用 Adam 作為優化器，更新模型的參數
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            optimizer.step()
        # 計算並顯示每個回合的平均損失值
        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch + 1}, Loss {avg_loss:.3f}")

# 定義一個函數，用來使用模型生成摘要或重點
def generate_summary(model, text):
    # 將模型切換到生成模式
    model.eval()
    # 將輸入的文本轉換成模型可以接受的格式
    input_ids = t5_preprocess(text)
    # 將輸入的張量放到 GPU 上，如果有的話
    input_ids = input_ids.to("cuda") if torch.cuda.is_available() else input_ids
    # 使用模型的 generate 方法來生成輸出的張量
    # 我們指定了以下的參數：
    # input_ids: 輸入的張量，包含了文本的數字表示
    # max_length: 生成的文本的最大長度，這裡是 150
    # num_beams: 生成的文本的候選數量，這裡是 5，也就是會生成 5 個不同的文本，並從中選擇最佳的一個
    output_ids = model.generate(input_ids, max_length=150, num_beams=5)
    # 將輸出的張量轉換成文本格式
    output_text = t5_postprocess(output_ids[0])
    # 回傳生成的文本
    return output_text

# 呼叫 train_model 函數，並傳入模型、資料載入器和訓練回合數，這裡我們設定訓練回合數為 3
train_model(model, dl, epochs=3)

#呼叫 generate_summary 函數，並傳入模型和一個文本資料，這裡我們使用資料集中的第一個文本作為範例
text = dset[0] 
summary = generate_summary(model, text)

顯示原始的文本和生成的摘要或重點
print(f"原始文本: {text}") 
print(f"生成摘要: {summary}")