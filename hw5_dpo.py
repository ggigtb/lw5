# -*- coding: utf-8 -*-
"""
SmolVLM-256M QLoRA微调 + DPO偏好对齐——实验作业五参考框架
对应教材「参数高效微调」与「偏好对齐/DPO」相关章节
单文件、开箱即跑：内置PIL合成「几何图形」图像15张、问答15题、
SFT样本15条、偏好对12组。不配GPU也能完成基线推理与小规模演示
（CPU上训练极慢，仅建议各跑5~10步体验流程；正式训练用Colab T4）。

常用命令：
python hw5_dpo.py --mode make_data   # 生成15张合成图像（首次运行自动执行）
python hw5_dpo.py --mode infer       # 基线推理15题，记录基线正确率
python hw5_dpo.py --mode sft         # QLoRA SFT（产出 ./sft_adapter）
python hw5_dpo.py --mode dpo         # 在SFT之上做DPO（产出 ./dpo_adapter）
python hw5_dpo.py --mode eval        # 三阶段（基线/SFT/SFT+DPO）× 15题对比
python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1   # DPO loss手算复核
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMG_DIR = "hw5_imgs"
SFT_DIR, DPO_DIR = "sft_adapter", "dpo_adapter"

# ============================================================
# 1. 合成图像库「几何图形」：15张图，三类问题的答案均已埋入
#    每张图 = 白底256x256，画若干「颜色-形状」对（位置固定可复现）
# ============================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}

# img_id -> [(颜色, 形状, 个数), ...]（形状：圆=ellipse，方=rectangle）
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],                      # 判断题素材：只有一个黄方
    "img12": [("红", "方", 1), ("蓝", "方", 1)],      # 没有圆形 → 「有没有红圆」答「没有」
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}


def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/")


def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")


# ============================================================
# 2. 问题集：3类 × 5题（gold = 标准答案；与参考答案逐字匹配计分）
#    count=数数（局部事实）attr=属性识别 exist=存在性/组合判断
# ============================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]

# SFT数据：图像 + 问题 → 简短标准答案（教模型「答题格式 + 事实」）
SFT_DATA = [{"img": q["img"], "prompt": q["q"],
             "answer": f"{q['gold']}。" if q["type"] != "count" else f"{q['gold']}个。"}
            for q in QUESTIONS]

# 偏好对（12组）：chosen = 正确回答；rejected = 典型幻觉/错误
# rejected的三类错误：数错（off-by-one/混淆）、形状幻觉、对不存在的物体硬答「有」
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？",
     "chosen": "2个。", "rejected": "3个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3个。", "rejected": "2个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？",
     "chosen": "2个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4个。", "rejected": "3个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？",
     "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
]


# ============================================================
# 3. 模型加载：QLoRA（教材：NF4量化 + LoRA低秩适配器）
#    256M小模型按理可全参微调，但本实验统一走QLoRA流程以体会方法
# ============================================================
def load_model(mode="infer"):
    """mode: infer=原精度推理 / train=4bit量化+LoRA（CPU无bitsandbytes时自动回退fp32+LoRA）"""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    kwargs = {"torch_dtype": torch.float32}
    use_4bit = False
    if mode == "train":
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32)
            use_4bit = True
        except ImportError:
            print("[load] 未检测到 bitsandbytes，自动回退为 fp32 基座 + LoRA")
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **kwargs)
    if mode == "train":
        from peft import LoraConfig, get_peft_model
        if use_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()   # ← 报告表1要抄的可训练参数占比
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def build_prompt(processor, q):
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    model.eval()   # 确保 dropout 等训练态算子关闭，逐题输出可复现
    inputs = processor(text=[build_prompt(processor, q)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.strip()


def judge(q, pred):
    """宽松判分：标准答案出现在预测里即算对（「没有」须整词出现）"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p and "有" in p
    return g in p


# ============================================================
# 4. SFT（教材：监督微调）：只对答案token计损失，全流程手写便于理解
# ============================================================
def sft_step(model, processor, opt, epochs=2, steps=60, log_every=10):
    dev = next(model.parameters()).device
    model.train()
    done = False
    for ep in range(epochs):
        if done:
            break
        rnd = random.Random(ep)
        order = list(range(len(SFT_DATA)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[sft] 已达 {steps} 步，提前结束")
                done = True
                break
            s = SFT_DATA[idx]
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[load_image(s["img"])],
                             return_tensors="pt").to(dev)
            with torch.no_grad():
                prompt_len = processor(text=[build_prompt(processor, s["prompt"])],
                                       images=[load_image(s["img"])],
                                       return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100   # 只学答案
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (i + 1) % log_every == 0:
                print(f"[sft] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f}")
    model.save_pretrained(SFT_DIR)
    print(f"[sft] 适配器已保存到 ./{SFT_DIR}")


# ============================================================
# 5. DPO（教材：直接偏好优化）
#    loss = -log σ( β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] )
#    参考模型 = 关闭LoRA适配器后的基座（disable_adapter上下文）
# ============================================================
def seq_logprob(model, processor, img, prompt, resp):
    """返回「答案token的平均对数概率 × 长度」= 序列总对数概率"""
    dev = next(model.parameters()).device
    full = processor(text=[build_prompt(processor, prompt) + resp],
                     images=[img], return_tensors="pt").to(dev)
    plen = processor(text=[build_prompt(processor, prompt)], images=[img],
                     return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    logits = model(**full).logits[:, :-1, :]
    tgt = labels[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = tgt.ne(-100)
    return lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum()


def dpo_step(model, processor, opt, beta=0.1, epochs=1, steps=40, log_every=5):
    dev = next(model.parameters()).device
    model.train()
    done = False
    for ep in range(epochs):
        if done:
            break
        rnd = random.Random(ep)
        order = list(range(len(PRE_PAIRS)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[dpo] 已达 {steps} 步，提前结束")
                done = True
                break
            p = PRE_PAIRS[idx]
            img = load_image(p["img"])
            pol_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
            pol_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            with model.disable_adapter():   # 冻结的参考模型 π_ref
                ref_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
                ref_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (i + 1) % log_every == 0:
                print(f"[dpo] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f} "
                      f"margin={margin.item():.3f}")
    model.save_pretrained(DPO_DIR)
    print(f"[dpo] 适配器已保存到 ./{DPO_DIR}")


def dpo_check(arg):
    """手算复核：依次给 pol_w, ref_w, pol_l, ref_l, beta"""
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    margin = beta * ((pw - rw) - (pl - rl))
    # -log σ(margin) = log(1 + e^{-margin})，用一个分支数值稳定地算，结果必为非负
    loss = math.log(1 + math.exp(-margin)) if margin >= 0 else -margin + math.log(1 + math.exp(margin))
    print(f"margin = β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] = {margin:.4f}")
    print(f"DPO loss = -log σ(margin) = {loss:.4f}"
          f"（margin越大loss越小；margin<0说明模型更偏好rejected，需要继续训练）")


# ============================================================
# 6. 三阶段对比评测：基线 / SFT / SFT+DPO × 15题（对照实验作业三阶段表）
# ============================================================
def evaluate(processor, model, tag, verbose=True):
    print(f"\n===== {tag} × 15 题 =====")
    t0 = time.time()
    ok_by_type = {}
    for tp in ("count", "attr", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = 0
        for q in qs:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            good = judge(q, pred)
            ok += good
            if verbose:   # 逐题原文，供报告 4.3 / 4.4 / 6.1 引用
                print(f"[{q['id']}·{q['type']}] {'√' if good else '×'} "
                      f"Q: {q['q']} A: {pred}（gold: {q['gold']}）")
        ok_by_type[tp] = (ok, len(qs))
        print(f"{tag:<10} {tp:<8} 正确 {ok}/{len(qs)}")
    total = sum(o for o, _ in ok_by_type.values())
    print(f"{tag:<10} 总计正确 {total}/15（本阶段15题耗时 {time.time()-t0:.1f}s）\n")
    return total


# ============================================================
# 7. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval"],
                    default="infer")
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数（CPU演示用5~10）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO温度系数β")
    ap.add_argument("--log_every", type=int, default=None, help="日志间隔（演示少步时设为1）")
    ap.add_argument("--dpo_check", default=None, help="如 1.2,0.8,0.9,0.6,0.1")
    args = ap.parse_args()

    if args.dpo_check:
        dpo_check(args.dpo_check)
        return

    if args.mode == "make_data":
        make_images()
        return

    if args.mode == "infer":
        processor, model = load_model("infer")
        for q in QUESTIONS:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            mark = "√" if judge(q, pred) else "×"
            print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q']} A: {pred}（gold: {q['gold']}）")
        return

    if args.mode == "sft":
        processor, model = load_model("train")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sft_step(model, processor, opt, steps=args.steps or 60,
                 log_every=args.log_every or 10)
        return

    if args.mode == "dpo":
        if not os.path.exists(SFT_DIR):
            raise SystemExit("先运行 --mode sft 生成 SFT 适配器")
        from peft import PeftModel
        processor, base = load_model("infer")   # 基座不加随机LoRA，直接在SFT适配器上续训DPO
        model = PeftModel.from_pretrained(base, SFT_DIR, is_trainable=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        dpo_step(model, processor, opt, beta=args.beta, steps=args.steps or 40,
                 log_every=args.log_every or 5)
        return

    if args.mode == "eval":
        # 三阶段共用一份基座，用 peft 多适配器 API 切换，避免重复加载 500MB 权重。
        # 注意：DPO_DIR 保存的是「在 SFT 之上继续训练」后的完整适配器权重，
        # 因此 SFT+DPO 阶段直接把该适配器挂到基座上，不能再叠加一次 SFT 适配器。
        processor, base = load_model("infer")
        evaluate(processor, base, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, SFT_DIR, adapter_name="sft")
            model.to(base.device)
            evaluate(processor, model, "SFT")
            if os.path.exists(DPO_DIR):
                model.load_adapter(DPO_DIR, adapter_name="dpo")
                model.set_adapter("dpo")
                evaluate(processor, model, "SFT+DPO")
        print("提示：把三阶段结果抄进报告表4/表5；另抽2题做人工忠实度评分（答案是否仅陈述图中事实）。")


if __name__ == "__main__":
    main()