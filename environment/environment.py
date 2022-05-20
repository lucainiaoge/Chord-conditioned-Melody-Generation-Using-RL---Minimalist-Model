"""Definition of the environment, state, and action."""

from typing import List
import copy
from pydantic import BaseModel

from music_struct import Note, Chord, TimeSignature, Tempo
from .environment_utils import sum_note_durations, find_state_id
import gym
from gym import spaces

class ComposerState(BaseModel):
    note_track: List[Note]
    beat_pos_track: List[float]
    time: int
    chord: Chord
    time_signature: TimeSignature
    tempo: Tempo
    beat_pos: float
    next_chord: Chord
    time_to_go: int
    max_time: int
    progress_percent: float
    done: bool
    n_step: int

ComposerAction = Note


class ComposerEnvironmentConditional(gym.Env):
    """Environment used in our project.

    Args:
        chords: a list of chords
        tempos: a list of tempos
        time_signatures: a list of time signatures
        first_downbeat: an integer indicating the advent of the first down beat
        init_note_track: a list in format like [{'pitch': 76, 'duration': 24}], indicating the initial notes
        title: a string, indicating the name of the piece
        resolution: an integer, indicating the ticks for one 4th note

    """

    def __init__(
        self,
        chords: List[Chord],
        tempos: List[Tempo],
        time_signatures: List[TimeSignature],
        first_downbeat: int = None,
        init_note_track: List[Note] = [],
        title: str = None,
        resolution: int = 24,
    ):
        self.title = title
        self.resolution = resolution

        self.chords = chords
        self.tempos = tempos
        self.time_signatures = time_signatures

        self.beat_length = int(resolution * 4 / time_signatures[0].denominator)
        self.measure_length = int(time_signatures[0].numerator * self.beat_length)

        if len(init_note_track) <= 0:
            self.note_track = []
            self.beat_pos_track = []
        else:
            self.note_track = copy.deepcopy(init_note_track)
            self.beat_pos_track = [-1]*len(self.note_track)

        if first_downbeat is None:
            self.first_downbeat = chords[0].time
        elif first_downbeat >= 0:
            self.first_downbeat = int(first_downbeat)
        else:
            raise TypeError("Incorrect format of initial downbeat.")

        self.time = 0
        self.chord_id = 0
        self.tempo_id = 0
        self.time_signature_id = 0
        self.n_step = 0
        self.max_time = chords[-1].time + chords[-1].duration
        # this may change the "current time or id", if init_note_track is not zero
        self.init_state = self.get_init_state(init_note_track)

    def reset(
        self,
        chords: List[Chord],
        tempos: List[Tempo],
        time_signatures: List[TimeSignature],
        first_downbeat=None,
        init_note_track=[],
        title=None,
        resolution=24,
    ):
        self.__init__(
            chords,
            tempos,
            time_signatures,
            first_downbeat=first_downbeat,
            init_note_track=init_note_track,
            title=title,
            resolution=resolution,
        )
        return self.init_state

    def step(self, action: ComposerAction) -> ComposerState:
        self.n_step += 1
        self.note_track.append(
            Note(
                time=self.time,
                pitch=action.pitch,
                duration=action.duration,
            )
        )

        self.time = self.time + action.duration

        # decide the next chord state
        if self.chord_id < len(self.chords) - 1:
            # first assume that the next chord state stays
            chord = self.chords[self.chord_id]
            next_chord = self.chords[self.chord_id + 1]
            time_to_go = next_chord.time - self.time

            # if chord does not stay, then go through the next chords until note falls into the scope
            while self.time >= next_chord.time:
                self.chord_id = self.chord_id + 1
                if self.chord_id < len(self.chords) - 1:
                    chord = self.chords[self.chord_id]
                    next_chord = self.chords[self.chord_id + 1]
                    time_to_go = next_chord.time - self.time
                else:
                    chord = self.chords[-1]
                    next_chord = self.chords[-1]
                    time_to_go = 0
                    break
        else:
            chord = self.chords[-1]
            next_chord = self.chords[-1]
            time_to_go = 0
        # decide the next tempo state
        if self.tempo_id < len(self.tempos) - 1:
            tempo_time = self.tempos[self.tempo_id + 1].time
            next_tempo = self.tempos[self.tempo_id]
            while self.time >= tempo_time:
                self.tempo_id = self.tempo_id + 1
                if self.tempo_id < len(self.tempos) - 1:
                    next_tempo = self.tempos[self.tempo_id]
                    tempo_time = self.tempos[self.tempo_id + 1].time
                else:
                    next_tempo = self.tempos[-1]
                    tempo_time = self.time
                    break
        else:
            next_tempo = self.tempos[-1]

        # decide if the next time is a down beat
        if self.time_signature_id < len(self.time_signatures) - 1:
            time_signature_time = self.time_signatures[self.time_signature_id + 1].time
            next_time_signature = self.time_signatures[self.time_signature_id]
            while self.time >= time_signature_time:
                self.time_signature_id = self.time_signature_id + 1
                if self.time_signature_id < len(self.time_signatures) - 1:
                    next_time_signature = self.time_signatures[self.time_signature_id]
                    time_signature_time = self.time_signatures[
                        self.time_signature_id + 1
                    ].time
                else:
                    next_time_signature = self.time_signatures[-1]
                    time_signature_time = self.time
                    break
            # when time signature has changed, we have to re-calculate the beat length and measure length
            self.beat_length = int(
                self.resolution * 4 / next_time_signature.denominator
            )
            self.measure_length = int(next_time_signature.numerator * self.beat_length)
        else:
            next_time_signature = self.time_signatures[-1]

        if self.time_signature_id > 0:
            self.first_downbeat = self.time_signatures[self.time_signature_id].time

        unbiased_time_elapse = self.time - self.first_downbeat
        beat_pos = (unbiased_time_elapse % self.measure_length) / self.measure_length

        done = self.time >= self.max_time

        progress_percent = self.time / self.max_time
        
        self.beat_pos_track.append(beat_pos)

        state_dict = {
            "note_track": self.note_track,
            "beat_pos_track": self.beat_pos_track, 
            "time": self.time,
            "chord": chord,
            "tempo": next_tempo,
            "time_signature": next_time_signature,
            "beat_pos": beat_pos,
            "next_chord": next_chord,
            "time_to_go": time_to_go,
            "max_time": self.max_time,
            "progress_percent": progress_percent,
            "done": done,
            "n_step": self.n_step,
        }

        return ComposerState(**state_dict)

    def get_init_state(self, init_note_track=[]) -> ComposerState:
        self.n_step = 0
        if len(init_note_track) <= 0:
            self.time = 0
            self.chord_id = 0
            self.tempo_id = 0
            self.time_signature_id = 0
        else:
            self.time = sum_note_durations(init_note_track)
            self.chord_id = find_state_id(self.time, self.chords)
            self.tempo_id = find_state_id(self.time, self.tempos)
            self.time_signature_id = find_state_id(
                self.time_signature_id, self.time_signatures
            )
            if self.time_signature_id > 0:
                self.beat_length = int(
                    self.resolution
                    * 4
                    / self.time_signatures[self.time_signature_id].denominator
                )
                self.measure_length = int(
                    self.time_signatures[self.time_signature_id].numerator
                    * self.beat_length
                )
                self.first_downbeat = self.time_signatures[self.time_signature_id].time

        chord = self.chords[self.chord_id]
        if self.chord_id < len(self.chords) - 1:
            next_chord = self.chords[self.chord_id + 1]
            time_to_go = next_chord.time - self.time
        else:
            next_chord = self.chords[-1]
            time_to_go = 0

        tempo = self.tempos[self.tempo_id]
        time_signature = self.time_signatures[self.time_signature_id]
        unbiased_time_elapse = self.time - self.first_downbeat
        beat_pos = (unbiased_time_elapse % self.measure_length) / self.measure_length
        self.beat_pos_track = [beat_pos]*(len(self.note_track) + 1)

        done = self.time >= self.max_time

        progress_percent = self.time / self.max_time

        state_dict = {
            "note_track": self.note_track,
            "beat_pos_track": self.beat_pos_track,
            "time": self.time,
            "chord": chord,
            "tempo": tempo,
            "time_signature": time_signature,
            "beat_pos": beat_pos,
            "next_chord": next_chord,
            "time_to_go": time_to_go,
            "max_time": self.max_time,
            "progress_percent": progress_percent,
            "done": done,
            "n_step": self.n_step,
        }

        return ComposerState(**state_dict)
