import RPi.GPIO as GPIO
import time

# 蜂鸣器连接的GPIO口
buzzer_pin = 23

# 使用BCM编号方式
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

# 初始化PWM（初始频率可任意设置）
pwm = GPIO.PWM(buzzer_pin, 440)
pwm.start(0)  # 初始静音

# 定义音符对应的频率
notes = {
    'do': 261,       # C4
    're': 293,       # D4
    'mi': 329,       # E4
    'fa': 349,       # F4
    'so': 392,       # G4
    'la': 440,       # A4
    'si': 493,       # B4
    'do_high': 523   # C5
}

def play_melody(state, note_duration=0.05, pause=0.05):
    """
    根据state播放音效：
      - state="success" 播放 ['do', 're', 'mi', 'so']
      - state="fail" 播放 ['so', 'mi', 're', 'do']
    """
    if state == "success":
        melody = ['do', 're', 'mi', 'so']
    elif state == "fail":
        melody = ['so', 'mi', 're', 'do']
    else:
        print("未知状态")
        return
    
    for note in melody:
        if note in notes:
            freq = notes[note]
            pwm.ChangeFrequency(freq)
            pwm.ChangeDutyCycle(50)  # 发声
        else:
            pwm.ChangeDutyCycle(0)   # 未定义音符时静音
        time.sleep(note_duration)
        # 每个音符播放后短暂静音
        pwm.ChangeDutyCycle(0)
        time.sleep(pause)
    # pwm.stop()
    # GPIO.cleanup()

#     # 播放成功音效
# print("播放成功音效")
# play_melody("success")

# time.sleep(1)  # 等待1秒

# # 播放失败音效
# print("播放失败音效")
# play_melody("fail")

