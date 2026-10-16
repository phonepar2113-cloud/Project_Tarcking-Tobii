#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on February 05, 2026, at 13:27
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from init
import tobii_research as tr
import time
import math
from datetime import datetime
import os
import csv
import pandas as pd

use_frequency = 150.0


# ========== STATE ==========
gaze_left_eye = None
gaze_right_eye = None
gaze_data_log = []

shared_fixation = {
    'active': False,
    'target': None  # ต้องเป็น object ที่มี .contains(x, y)
}

fixation_state = {
    'entered': False,
    'start_time': None,
    'sustained': False
}

# ใช้สำหรับเก็บชื่อหน้าปัจจุบัน และ state ล่าสุดที่ถูก log
gaze_state = {
    'current': None,
    'last_logged': None
}


# ========== TOBII SETUP ==========
found_eyetrackers = tr.find_all_eyetrackers()
if not found_eyetrackers:
    raise RuntimeError("No Tobii eye tracker found.")
my_eyetracker = found_eyetrackers[0]
print(f"Connected to: {my_eyetracker}")

freqs = my_eyetracker.get_all_gaze_output_frequencies()
print(f"Available frequencies: {freqs}")
if use_frequency in freqs:
    my_eyetracker.set_gaze_output_frequency(use_frequency)
    current_freq = my_eyetracker.get_gaze_output_frequency()
    print(f"Sampling rate currently in use: {current_freq} Hz")
else:
    print(f"Warning: {use_frequency} Hz not supported.")


# ========== TIMESTAMP HELPERS ==========
def get_timestamp_raw():
    return time.time()

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

# ========== UNIT CONVERSION ==========
def convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm):
    gaze_x_psy = (gaze_x_norm - 0.5) * 2
    gaze_y_psy = (0.5 - gaze_y_norm) * 2
    return gaze_x_psy, gaze_y_psy

# ========== GAZE CALLBACK ==========
def gaze_data_callback(gaze_data):
    global gaze_left_eye, gaze_right_eye, fixation_state

    gaze_left_eye = gaze_data['left_gaze_point_on_display_area']
    gaze_right_eye = gaze_data['right_gaze_point_on_display_area']

    try:
        gaze_x_norm, gaze_y_norm = gaze_right_eye
        if math.isnan(gaze_x_norm) or math.isnan(gaze_y_norm):
            return

        gaze_x, gaze_y = convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm)

        # === LOG GAZE DATA ===
        # ใส่ state เฉพาะบรรทัดแรกของแต่ละ routine
        if gaze_state['current'] != gaze_state['last_logged']:
            current_state_label = gaze_state['current'] or ""
            gaze_state['last_logged'] = gaze_state['current']
        else:
            current_state_label = ""

        gaze_data_log.append({
            'timestamp_computer': get_timestamp(),
            'timestamp_computer_raw': get_timestamp_raw(),
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'state': current_state_label
            })
#        print(f"[DEBUG]gaze_data_log {gaze_data_log[-1]} ")

        # === FIXATION CHECK (ไม่เก็บ log) ===
        if shared_fixation['active'] and shared_fixation['target'] is not None:
            now = time.perf_counter()
            if shared_fixation['target'].contains(gaze_x, gaze_y):
#                print("Enter")
#                print(f"Position {gaze_x}, {gaze_y}")
                if not fixation_state['entered']:
                    fixation_state['entered'] = True
                    fixation_state['start_time'] = now
                    fixation_state['sustained'] = False
                elif now - fixation_state['start_time'] >= 1.0:
#                    print("Success")
                    fixation_state['sustained'] = True
            else:
#                print(2)
                fixation_state['entered'] = False
                fixation_state['start_time'] = None
                fixation_state['sustained'] = False

    except Exception as e:
        print(f"[ERROR] gaze_data_callback failed: {e}")

# ========== START CALLBACK ==========
my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)


# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'try_font_reading_task'  # from the Builder filename that created this script
expInfo = {
    'participant': '9999',
    'session': '01',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u"data/p%s_s%s_%s" % (expInfo['participant'], str(expInfo['session']).zfill(2), expName)
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\Font project 2026\\1-Reading Main\\font_reading_task.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.WARNING)


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='ptb')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='PsychToolbox')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instruction_routine" ---
    inst_img = visual.ImageStim(
        win=win,
        name='inst_img', units='norm', 
        image='stimulus/instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_inst = keyboard.Keyboard()
    # Run 'Begin Experiment' code from init
    screen_width, screen_height = win.size
    # Normalize session to two digits (allow "1" or "01")
    sess_raw = str(expInfo.get('session', '')).strip()
    sessionStr = sess_raw.zfill(2)
    
    # Build the session file path (adjust folder/name to your actual files)
    # Example assumes: ./sessions/session01.xlsx ... session18.xlsx
    sessionFile = os.path.join('session_file', f'session{sessionStr}.xlsx')
    
    # (Optional) sanity check in console
    if not os.path.exists(sessionFile):
        print(f"WARNING: Session file not found: {sessionFile}")
    
    
    participant_id = expInfo['participant']
    session_id = expInfo['session']
    
    os.makedirs("gaze_data", exist_ok=True)
    
    base_name = f"p{participant_id}_s{session_id}_gaze_data"
    ext = ".csv"
    
    # ---- find available filename ----
    counter = 0
    while True:
        if counter == 0:
            gaze_csv_path = os.path.join("gaze_data", base_name + ext)
        else:
            gaze_csv_path = os.path.join("gaze_data", f"{base_name}_{counter}{ext}")
    
        if not os.path.exists(gaze_csv_path):
            break
        counter += 1
    
    # ---- create file + write header once ----
    with open(gaze_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_computer",
            "timestamp_computer_raw",
            "gaze_x",
            "gaze_y",
            "state"
        ])
    
    gaze_header_written = True
    
    print(f"[GAZE] Writing to: {gaze_csv_path}")
    
    # --- Initialize components for Routine "fixation_routine" ---
    fixation_img = visual.ImageStim(
        win=win,
        name='fixation_img', units='norm', 
        image='stimulus/fixation.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=0.5,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    fixation_area = visual.ShapeStim(
        win=win, name='fixation_area',
        size=(0.5, 0.5), vertices='circle',
        ori=0.0, pos=(-0.75, 0.52), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.2941, -0.6706, -0.6706],
        opacity=0.0, depth=-1.0, interpolate=True)
    # Run 'Begin Experiment' code from code_4
    
    
    # Create gaze dot and text for visualization
    screen_width, screen_height = win.size
    gaze_text = visual.TextStim(
        win,
        text='',
        pos=(0.6, 0.45),
        height=0.05,
        color='white',
        colorSpace='rgb',
        units='norm',
        name='gaze_text',
        autoLog=False
    )
    
    gaze_dot = visual.Circle(
        win,
        radius=0.02,
        fillColor='red',
        lineColor='red',
        pos=(0, 0),
        units='norm',
        name='gaze_dot',
        autoLog=False
    )
    
    
    # --- Initialize components for Routine "reading_routine" ---
    reading_img = visual.ImageStim(
        win=win,
        name='reading_img', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_finish_read = keyboard.Keyboard()
    
    # --- Initialize components for Routine "question_routine" ---
    question_image = visual.ImageStim(
        win=win,
        name='question_image', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_answer = keyboard.Keyboard()
    
    # --- Initialize components for Routine "ratingcomp_routine" ---
    ratingcomp_img = visual.ImageStim(
        win=win,
        name='ratingcomp_img', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_ratingcomp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "ratingfont_routine" ---
    ratingfont_img = visual.ImageStim(
        win=win,
        name='ratingfont_img', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_ratingfont = keyboard.Keyboard()
    
    # --- Initialize components for Routine "end_routine" ---
    end_img = visual.ImageStim(
        win=win,
        name='end_img', units='norm', 
        image='stimulus/end_exp.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "instruction_routine" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_routine.started', globalClock.getTime())
    key_resp_inst.keys = []
    key_resp_inst.rt = []
    _key_resp_inst_allKeys = []
    # Run 'Begin Routine' code from init
    gaze_state['current'] = 'begin'
    # keep track of which components have finished
    instruction_routineComponents = [inst_img, key_resp_inst]
    for thisComponent in instruction_routineComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction_routine" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *inst_img* updates
        
        # if inst_img is starting this frame...
        if inst_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            inst_img.frameNStart = frameN  # exact frame index
            inst_img.tStart = t  # local t and not account for scr refresh
            inst_img.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(inst_img, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'inst_img.started')
            # update status
            inst_img.status = STARTED
            inst_img.setAutoDraw(True)
        
        # if inst_img is active this frame...
        if inst_img.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst* updates
        waitOnFlip = False
        
        # if key_resp_inst is starting this frame...
        if key_resp_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst.frameNStart = frameN  # exact frame index
            key_resp_inst.tStart = t  # local t and not account for scr refresh
            key_resp_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_inst.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_allKeys.extend(theseKeys)
            if len(_key_resp_inst_allKeys):
                key_resp_inst.keys = _key_resp_inst_allKeys[0].name  # just the first key pressed
                key_resp_inst.rt = _key_resp_inst_allKeys[0].rt
                key_resp_inst.duration = _key_resp_inst_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruction_routineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_routine" ---
    for thisComponent in instruction_routineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction_routine.stopped', globalClock.getTime())
    # check responses
    if key_resp_inst.keys in ['', [], None]:  # No response was made
        key_resp_inst.keys = None
    thisExp.addData('key_resp_inst.keys',key_resp_inst.keys)
    if key_resp_inst.keys != None:  # we had a response
        thisExp.addData('key_resp_inst.rt', key_resp_inst.rt)
        thisExp.addData('key_resp_inst.duration', key_resp_inst.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from init
    
    # กันไว้ เผื่อยังไม่ set path
    if gaze_csv_path and len(gaze_data_log) > 0:
    #    routine_name = gaze_state['current'] or ""  # หรือใช้ชื่อ routine จริงถ้าคุณ set ไว้
    
        with open(gaze_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in gaze_data_log:
                writer.writerow([
                    row.get("timestamp_computer", ""),
                    row.get("timestamp_computer_raw", ""),
                    row.get("gaze_x", ""),
                    row.get("gaze_y", ""),
                    row.get("state", "")
                ])
    
    # ล้าง buffer เพื่อไม่กิน RAM
    gaze_data_log.clear()
    # รีเซ็ต gaze state เพื่อไม่ให้ log state ซ้ำ
    gaze_state['current'] = None
    # the Routine "instruction_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loopexp = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(sessionFile),
        seed=None, name='loopexp')
    thisExp.addLoop(loopexp)  # add the loop to the experiment
    thisLoopexp = loopexp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoopexp.rgb)
    if thisLoopexp != None:
        for paramName in thisLoopexp:
            globals()[paramName] = thisLoopexp[paramName]
    
    for thisLoopexp in loopexp:
        currentLoop = loopexp
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoopexp.rgb)
        if thisLoopexp != None:
            for paramName in thisLoopexp:
                globals()[paramName] = thisLoopexp[paramName]
        
        # --- Prepare to start Routine "fixation_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation_routine.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_4
        # === Fixation Begin Routine ===
        
        # แจ้งระบบ logger ว่าตอนนี้อยู่หน้า fixation (จะใช้กับ gaze_state['current'])
        gaze_state['current'] = 'fixation'
        
        # รีเซ็ตสถานะ fixation เพื่อเริ่มตรวจจับใหม่
        fixation_state['entered'] = False
        fixation_state['start_time'] = None
        fixation_state['sustained'] = False
        
        # กำหนดพื้นที่ fixation ที่ต้องการให้ผู้เข้าร่วมจ้อง
        shared_fixation['active'] = True
        shared_fixation['target'] = fixation_area  # fixation_area ควรเป็น visual.Rect หรือ visual.Circle ที่มี .contains(x, y)
        
        # keep track of which components have finished
        fixation_routineComponents = [fixation_img, fixation_area]
        for thisComponent in fixation_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_img* updates
            
            # if fixation_img is starting this frame...
            if fixation_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_img.frameNStart = frameN  # exact frame index
                fixation_img.tStart = t  # local t and not account for scr refresh
                fixation_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_img.started')
                # update status
                fixation_img.status = STARTED
                fixation_img.setAutoDraw(True)
            
            # if fixation_img is active this frame...
            if fixation_img.status == STARTED:
                # update params
                pass
            
            # *fixation_area* updates
            
            # if fixation_area is starting this frame...
            if fixation_area.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_area.frameNStart = frameN  # exact frame index
                fixation_area.tStart = t  # local t and not account for scr refresh
                fixation_area.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_area, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_area.started')
                # update status
                fixation_area.status = STARTED
                fixation_area.setAutoDraw(True)
            
            # if fixation_area is active this frame...
            if fixation_area.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_4
            # === Fixation Each Frame ===
            
            # เช็กว่า fixation ครบหรือยัง
            if fixation_state['sustained']:
                print("\n---------------Fixation complete — exiting fixation_screen.")
                shared_fixation['active'] = False
                shared_fixation['target'] = None
                gaze_state['current'] = None  # reset gaze state เพื่อไม่ให้ log state ซ้ำ
                continueRoutine = False
            
            # Optional: แสดงตำแหน่ง gaze
            if gaze_right_eye and not fixation_state['sustained']:
                try:
                    gaze_x_norm, gaze_y_norm = gaze_right_eye
                    if not math.isnan(gaze_x_norm) and not math.isnan(gaze_y_norm):
                        gaze_x, gaze_y = convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm)
                        gaze_dot.pos = (gaze_x, gaze_y)
                        gaze_dot.draw()
                        gaze_text.text = f"x: {gaze_x:.2f}, y: {gaze_y:.2f}"
                        gaze_text.draw()
                except Exception as e:
                    print(f"[WARNING] Failed to draw gaze dot: {e}")
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_routine" ---
        for thisComponent in fixation_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation_routine.stopped', globalClock.getTime())
        # Run 'End Routine' code from code_4
        # === Fixation End Routine ===
        # กันไว้ เผื่อยังไม่ set path
        if gaze_csv_path and len(gaze_data_log) > 0:
            routine_name = gaze_state['current'] or ""  # หรือใช้ชื่อ routine จริงถ้าคุณ set ไว้
        
            with open(gaze_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in gaze_data_log:
                    writer.writerow([
                        row.get("timestamp_computer", ""),
                        row.get("timestamp_computer_raw", ""),
                        row.get("gaze_x", ""),
                        row.get("gaze_y", ""),
                        row.get("state", "")
                    ])
        
        # ล้าง buffer เพื่อไม่กิน RAM
        gaze_data_log.clear()
        # ปิดระบบตรวจจับ fixation
        shared_fixation['active'] = False
        shared_fixation['target'] = None
        
        # รีเซ็ต gaze state เพื่อไม่ให้ log state ซ้ำ
        gaze_state['current'] = None
        
        # the Routine "fixation_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "reading_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('reading_routine.started', globalClock.getTime())
        reading_img.setImage(stimulus_img)
        key_finish_read.keys = []
        key_finish_read.rt = []
        _key_finish_read_allKeys = []
        # Run 'Begin Routine' code from code_5
        # === Trial Begin Routine ===
        gaze_state['current'] = 'reading'
        
        # keep track of which components have finished
        reading_routineComponents = [reading_img, key_finish_read]
        for thisComponent in reading_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reading_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 60.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *reading_img* updates
            
            # if reading_img is starting this frame...
            if reading_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reading_img.frameNStart = frameN  # exact frame index
                reading_img.tStart = t  # local t and not account for scr refresh
                reading_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reading_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reading_img.started')
                # update status
                reading_img.status = STARTED
                reading_img.setAutoDraw(True)
            
            # if reading_img is active this frame...
            if reading_img.status == STARTED:
                # update params
                pass
            
            # if reading_img is stopping this frame...
            if reading_img.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reading_img.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    reading_img.tStop = t  # not accounting for scr refresh
                    reading_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reading_img.stopped')
                    # update status
                    reading_img.status = FINISHED
                    reading_img.setAutoDraw(False)
            
            # *key_finish_read* updates
            waitOnFlip = False
            
            # if key_finish_read is starting this frame...
            if key_finish_read.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_finish_read.frameNStart = frameN  # exact frame index
                key_finish_read.tStart = t  # local t and not account for scr refresh
                key_finish_read.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_finish_read, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_finish_read.started')
                # update status
                key_finish_read.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_finish_read.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_finish_read.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_finish_read is stopping this frame...
            if key_finish_read.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_finish_read.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    key_finish_read.tStop = t  # not accounting for scr refresh
                    key_finish_read.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_finish_read.stopped')
                    # update status
                    key_finish_read.status = FINISHED
                    key_finish_read.status = FINISHED
            if key_finish_read.status == STARTED and not waitOnFlip:
                theseKeys = key_finish_read.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_finish_read_allKeys.extend(theseKeys)
                if len(_key_finish_read_allKeys):
                    key_finish_read.keys = _key_finish_read_allKeys[0].name  # just the first key pressed
                    key_finish_read.rt = _key_finish_read_allKeys[0].rt
                    key_finish_read.duration = _key_finish_read_allKeys[0].duration
            # Run 'Each Frame' code from code_5
            # === Trial Each Frame ===
            
            # วาดตำแหน่ง gaze แบบ real-time
            if gaze_right_eye:
                try:
                    gaze_x_norm, gaze_y_norm = gaze_right_eye
                    if not math.isnan(gaze_x_norm) and not math.isnan(gaze_y_norm):
                        gaze_x, gaze_y = convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm)
            #            gaze_dot.pos = (gaze_x, gaze_y)
            #            gaze_dot.draw()
            #            gaze_text.text = f"x: {gaze_x:.2f}, y: {gaze_y:.2f}"
            #            gaze_text.draw()
                except Exception as e:
                    print(f"[WARNING] Failed to draw gaze dot: {e}")
            
            # -------------------------------
            
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reading_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reading_routine" ---
        for thisComponent in reading_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('reading_routine.stopped', globalClock.getTime())
        # check responses
        if key_finish_read.keys in ['', [], None]:  # No response was made
            key_finish_read.keys = None
        loopexp.addData('key_finish_read.keys',key_finish_read.keys)
        if key_finish_read.keys != None:  # we had a response
            loopexp.addData('key_finish_read.rt', key_finish_read.rt)
            loopexp.addData('key_finish_read.duration', key_finish_read.duration)
        # Run 'End Routine' code from code_5
        # -------------------------------
        # กันไว้ เผื่อยังไม่ set path
        if gaze_csv_path and len(gaze_data_log) > 0:
            routine_name = gaze_state['current'] or ""  # หรือใช้ชื่อ routine จริงถ้าคุณ set ไว้
        
            with open(gaze_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in gaze_data_log:
                    writer.writerow([
                        row.get("timestamp_computer", ""),
                        row.get("timestamp_computer_raw", ""),
                        row.get("gaze_x", ""),
                        row.get("gaze_y", ""),
                        row.get("state", "")
                    ])
        
        # ล้าง buffer เพื่อไม่กิน RAM
        gaze_data_log.clear()
        gaze_state['current'] = None
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-60.000000)
        
        # --- Prepare to start Routine "question_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('question_routine.started', globalClock.getTime())
        question_image.setImage(question_img)
        key_answer.keys = []
        key_answer.rt = []
        _key_answer_allKeys = []
        # Run 'Begin Routine' code from code_6
        # === Trial Begin Routine ===
        gaze_state['current'] = 'question'
        
        # keep track of which components have finished
        question_routineComponents = [question_image, key_answer]
        for thisComponent in question_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_image* updates
            
            # if question_image is starting this frame...
            if question_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_image.frameNStart = frameN  # exact frame index
                question_image.tStart = t  # local t and not account for scr refresh
                question_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_image.started')
                # update status
                question_image.status = STARTED
                question_image.setAutoDraw(True)
            
            # if question_image is active this frame...
            if question_image.status == STARTED:
                # update params
                pass
            
            # *key_answer* updates
            waitOnFlip = False
            
            # if key_answer is starting this frame...
            if key_answer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_answer.frameNStart = frameN  # exact frame index
                key_answer.tStart = t  # local t and not account for scr refresh
                key_answer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_answer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_answer.started')
                # update status
                key_answer.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_answer.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_answer.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_answer.status == STARTED and not waitOnFlip:
                theseKeys = key_answer.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_answer_allKeys.extend(theseKeys)
                if len(_key_answer_allKeys):
                    key_answer.keys = _key_answer_allKeys[0].name  # just the first key pressed
                    key_answer.rt = _key_answer_allKeys[0].rt
                    key_answer.duration = _key_answer_allKeys[0].duration
                    # was this correct?
                    if (key_answer.keys == str(correct_key)) or (key_answer.keys == correct_key):
                        key_answer.corr = 1
                    else:
                        key_answer.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_6
            # === Trial Each Frame ===
            
            # วาดตำแหน่ง gaze แบบ real-time
            if gaze_right_eye:
                try:
                    gaze_x_norm, gaze_y_norm = gaze_right_eye
                    if not math.isnan(gaze_x_norm) and not math.isnan(gaze_y_norm):
                        gaze_x, gaze_y = convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm)
            #            gaze_dot.pos = (gaze_x, gaze_y)
            #            gaze_dot.draw()
            #            gaze_text.text = f"x: {gaze_x:.2f}, y: {gaze_y:.2f}"
            #            gaze_text.draw()
                except Exception as e:
                    print(f"[WARNING] Failed to draw gaze dot: {e}")
            
            
            
            
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in question_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_routine" ---
        for thisComponent in question_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('question_routine.stopped', globalClock.getTime())
        # check responses
        if key_answer.keys in ['', [], None]:  # No response was made
            key_answer.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_answer.corr = 1;  # correct non-response
            else:
               key_answer.corr = 0;  # failed to respond (incorrectly)
        # store data for loopexp (TrialHandler)
        loopexp.addData('key_answer.keys',key_answer.keys)
        loopexp.addData('key_answer.corr', key_answer.corr)
        if key_answer.keys != None:  # we had a response
            loopexp.addData('key_answer.rt', key_answer.rt)
            loopexp.addData('key_answer.duration', key_answer.duration)
        # Run 'End Routine' code from code_6
        # -------------------------------
        # กันไว้ เผื่อยังไม่ set path
        if gaze_csv_path and len(gaze_data_log) > 0:
            routine_name = gaze_state['current'] or ""  # หรือใช้ชื่อ routine จริงถ้าคุณ set ไว้
        
            with open(gaze_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in gaze_data_log:
                    writer.writerow([
                        row.get("timestamp_computer", ""),
                        row.get("timestamp_computer_raw", ""),
                        row.get("gaze_x", ""),
                        row.get("gaze_y", ""),
                        row.get("state", "")
                    ])
        
        # ล้าง buffer เพื่อไม่กิน RAM
        gaze_data_log.clear()
        gaze_state['current'] = None
        # the Routine "question_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "ratingcomp_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ratingcomp_routine.started', globalClock.getTime())
        ratingcomp_img.setImage(rating_comp_img)
        key_ratingcomp.keys = []
        key_ratingcomp.rt = []
        _key_ratingcomp_allKeys = []
        # Run 'Begin Routine' code from code_7
        # === Trial Begin Routine ===
        gaze_state['current'] = 'ratingcomp'
        
        # Skip this rating if not requested
        if not doRating or rating_comp_img in [None, '', 'nan']:
            continueRoutine = False
        
        
        # keep track of which components have finished
        ratingcomp_routineComponents = [ratingcomp_img, key_ratingcomp]
        for thisComponent in ratingcomp_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ratingcomp_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ratingcomp_img* updates
            
            # if ratingcomp_img is starting this frame...
            if ratingcomp_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ratingcomp_img.frameNStart = frameN  # exact frame index
                ratingcomp_img.tStart = t  # local t and not account for scr refresh
                ratingcomp_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ratingcomp_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ratingcomp_img.started')
                # update status
                ratingcomp_img.status = STARTED
                ratingcomp_img.setAutoDraw(True)
            
            # if ratingcomp_img is active this frame...
            if ratingcomp_img.status == STARTED:
                # update params
                pass
            
            # *key_ratingcomp* updates
            waitOnFlip = False
            
            # if key_ratingcomp is starting this frame...
            if key_ratingcomp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_ratingcomp.frameNStart = frameN  # exact frame index
                key_ratingcomp.tStart = t  # local t and not account for scr refresh
                key_ratingcomp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_ratingcomp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_ratingcomp.started')
                # update status
                key_ratingcomp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_ratingcomp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_ratingcomp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_ratingcomp.status == STARTED and not waitOnFlip:
                theseKeys = key_ratingcomp.getKeys(keyList=['1','2','3','4','5'], ignoreKeys=["escape"], waitRelease=False)
                _key_ratingcomp_allKeys.extend(theseKeys)
                if len(_key_ratingcomp_allKeys):
                    key_ratingcomp.keys = _key_ratingcomp_allKeys[0].name  # just the first key pressed
                    key_ratingcomp.rt = _key_ratingcomp_allKeys[0].rt
                    key_ratingcomp.duration = _key_ratingcomp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_7
            # === Trial Each Frame ===
            
            # วาดตำแหน่ง gaze แบบ real-time
            if gaze_right_eye:
                try:
                    gaze_x_norm, gaze_y_norm = gaze_right_eye
                    if not math.isnan(gaze_x_norm) and not math.isnan(gaze_y_norm):
                        gaze_x, gaze_y = convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm)
            #            gaze_dot.pos = (gaze_x, gaze_y)
            #            gaze_dot.draw()
            #            gaze_text.text = f"x: {gaze_x:.2f}, y: {gaze_y:.2f}"
            #            gaze_text.draw()
                except Exception as e:
                    print(f"[WARNING] Failed to draw gaze dot: {e}")
            
            
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ratingcomp_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ratingcomp_routine" ---
        for thisComponent in ratingcomp_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ratingcomp_routine.stopped', globalClock.getTime())
        # check responses
        if key_ratingcomp.keys in ['', [], None]:  # No response was made
            key_ratingcomp.keys = None
        loopexp.addData('key_ratingcomp.keys',key_ratingcomp.keys)
        if key_ratingcomp.keys != None:  # we had a response
            loopexp.addData('key_ratingcomp.rt', key_ratingcomp.rt)
            loopexp.addData('key_ratingcomp.duration', key_ratingcomp.duration)
        # Run 'End Routine' code from code_7
        
        # -------------------------------
        # กันไว้ เผื่อยังไม่ set path
        if gaze_csv_path and len(gaze_data_log) > 0:
            routine_name = gaze_state['current'] or ""  # หรือใช้ชื่อ routine จริงถ้าคุณ set ไว้
        
            with open(gaze_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in gaze_data_log:
                    writer.writerow([
                        row.get("timestamp_computer", ""),
                        row.get("timestamp_computer_raw", ""),
                        row.get("gaze_x", ""),
                        row.get("gaze_y", ""),
                        row.get("state", "")
                    ])
        
        # ล้าง buffer เพื่อไม่กิน RAM
        gaze_data_log.clear()
        gaze_state['current'] = None
        # the Routine "ratingcomp_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "ratingfont_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ratingfont_routine.started', globalClock.getTime())
        ratingfont_img.setImage(rating_font_img)
        key_ratingfont.keys = []
        key_ratingfont.rt = []
        _key_ratingfont_allKeys = []
        # Run 'Begin Routine' code from code
        # === Trial Begin Routine ===
        gaze_state['current'] = 'ratingfont'
        
        if not doRating or rating_font_img in [None, '', 'nan']:
            continueRoutine = False
        
        
        # keep track of which components have finished
        ratingfont_routineComponents = [ratingfont_img, key_ratingfont]
        for thisComponent in ratingfont_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ratingfont_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ratingfont_img* updates
            
            # if ratingfont_img is starting this frame...
            if ratingfont_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ratingfont_img.frameNStart = frameN  # exact frame index
                ratingfont_img.tStart = t  # local t and not account for scr refresh
                ratingfont_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ratingfont_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ratingfont_img.started')
                # update status
                ratingfont_img.status = STARTED
                ratingfont_img.setAutoDraw(True)
            
            # if ratingfont_img is active this frame...
            if ratingfont_img.status == STARTED:
                # update params
                pass
            
            # *key_ratingfont* updates
            waitOnFlip = False
            
            # if key_ratingfont is starting this frame...
            if key_ratingfont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_ratingfont.frameNStart = frameN  # exact frame index
                key_ratingfont.tStart = t  # local t and not account for scr refresh
                key_ratingfont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_ratingfont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_ratingfont.started')
                # update status
                key_ratingfont.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_ratingfont.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_ratingfont.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_ratingfont.status == STARTED and not waitOnFlip:
                theseKeys = key_ratingfont.getKeys(keyList=['1','2','3','4','5'], ignoreKeys=["escape"], waitRelease=False)
                _key_ratingfont_allKeys.extend(theseKeys)
                if len(_key_ratingfont_allKeys):
                    key_ratingfont.keys = _key_ratingfont_allKeys[0].name  # just the first key pressed
                    key_ratingfont.rt = _key_ratingfont_allKeys[0].rt
                    key_ratingfont.duration = _key_ratingfont_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code
            # === Trial Each Frame ===
            
            # วาดตำแหน่ง gaze แบบ real-time
            if gaze_right_eye:
                try:
                    gaze_x_norm, gaze_y_norm = gaze_right_eye
                    if not math.isnan(gaze_x_norm) and not math.isnan(gaze_y_norm):
                        gaze_x, gaze_y = convert_Tobii_units_to_PsychoPy(gaze_x_norm, gaze_y_norm)
            #            gaze_dot.pos = (gaze_x, gaze_y)
            #            gaze_dot.draw()
            #            gaze_text.text = f"x: {gaze_x:.2f}, y: {gaze_y:.2f}"
            #            gaze_text.draw()
                except Exception as e:
                    print(f"[WARNING] Failed to draw gaze dot: {e}")
            
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ratingfont_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ratingfont_routine" ---
        for thisComponent in ratingfont_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ratingfont_routine.stopped', globalClock.getTime())
        # check responses
        if key_ratingfont.keys in ['', [], None]:  # No response was made
            key_ratingfont.keys = None
        loopexp.addData('key_ratingfont.keys',key_ratingfont.keys)
        if key_ratingfont.keys != None:  # we had a response
            loopexp.addData('key_ratingfont.rt', key_ratingfont.rt)
            loopexp.addData('key_ratingfont.duration', key_ratingfont.duration)
        # Run 'End Routine' code from code
        # -------------------------------
        # กันไว้ เผื่อยังไม่ set path
        if gaze_csv_path and len(gaze_data_log) > 0:
            routine_name = gaze_state['current'] or ""  # หรือใช้ชื่อ routine จริงถ้าคุณ set ไว้
        
            with open(gaze_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in gaze_data_log:
                    writer.writerow([
                        row.get("timestamp_computer", ""),
                        row.get("timestamp_computer_raw", ""),
                        row.get("gaze_x", ""),
                        row.get("gaze_y", ""),
                        row.get("state", "")
                    ])
        
        # ล้าง buffer เพื่อไม่กิน RAM
        gaze_data_log.clear()
        gaze_state['current'] = None
        # the Routine "ratingfont_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'loopexp'
    
    
    # --- Prepare to start Routine "end_routine" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_routine.started', globalClock.getTime())
    # keep track of which components have finished
    end_routineComponents = [end_img]
    for thisComponent in end_routineComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_routine" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_img* updates
        
        # if end_img is starting this frame...
        if end_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_img.frameNStart = frameN  # exact frame index
            end_img.tStart = t  # local t and not account for scr refresh
            end_img.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_img, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_img.started')
            # update status
            end_img.status = STARTED
            end_img.setAutoDraw(True)
        
        # if end_img is active this frame...
        if end_img.status == STARTED:
            # update params
            pass
        
        # if end_img is stopping this frame...
        if end_img.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_img.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                end_img.tStop = t  # not accounting for scr refresh
                end_img.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_img.stopped')
                # update status
                end_img.status = FINISHED
                end_img.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_routineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_routine" ---
    for thisComponent in end_routineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_routine.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    # Run 'End Experiment' code from init
    ## Unsubscribe
    #my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    #os.makedirs("gaze_data", exist_ok=True)
    #
    ## Save to CSV
    #participant_id = expInfo['participant']
    #session_id = expInfo['session']
    #output_path = f"gaze_data/p{participant_id}_s{session_id}_gaze_data.csv"
    #with open(output_path, 'w', newline='') as csvfile:
    #    fieldnames = gaze_data_log[0].keys()
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    writer.writeheader()
    #    writer.writerows(gaze_data_log)
    #
    #print("Saved gaze_log_full.csv.")
    # Unsubscribe
    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    print(f"Saved gaze data (streamed) to: {gaze_csv_path}")
    
    #os.makedirs("gaze_data", exist_ok=True)
    #
    ## Save to CSV using pandas
    #participant_id = expInfo['participant']
    #session_id = expInfo['session']
    #output_path = f"gaze_data/p{participant_id}_s{session_id}_gaze_data.csv"
    #
    ## Convert gaze_data_log (list of dicts) to DataFrame
    #df = pd.DataFrame(gaze_data_log)
    #
    ## Save with high float precision
    #df.to_csv(output_path, index=False, float_format="%.6f")
    #
    #print("Saved gaze_log_full.csv.")
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
