import { useEffect, useMemo, useRef, useState } from "react";
import { Music, Pause, Play } from "lucide-react";

interface AudioPreviewPlaceholderProps {
  src?: string;
  fileName?: string;
  label?: string;
  className?: string;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function AudioPreviewPlaceholder({
  src,
  fileName,
  label = "Audio Preview",
  className = "",
}: AudioPreviewPlaceholderProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [src]);

  const progress = duration > 0 ? Math.min(currentTime / duration, 1) : 0;
  const rangeBackground = useMemo(
    () =>
      `linear-gradient(to right, rgba(110,231,183,0.95) 0%, rgba(110,231,183,0.95) ${
        progress * 100
      }%, rgba(6,78,59,0.65) ${progress * 100}%, rgba(6,78,59,0.65) 100%)`,
    [progress]
  );

  const togglePlayback = async () => {
    const audio = audioRef.current;
    if (!audio || !src) return;
    if (audio.paused) {
      try {
        await audio.play();
        setIsPlaying(true);
      } catch (error) {
        console.warn("Audio preview playback failed:", error);
      }
    } else {
      audio.pause();
      setIsPlaying(false);
    }
  };

  const handleScrub = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextTime = Number(event.target.value);
    setCurrentTime(nextTime);
    if (audioRef.current) {
      audioRef.current.currentTime = nextTime;
    }
  };

  return (
    <div
      className={`flex h-full min-h-[72px] w-full flex-col justify-between rounded-md bg-emerald-400/[0.07] px-3 py-2 text-emerald-200 ${className}`}
    >
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={event => {
          const nextDuration = event.currentTarget.duration;
          setDuration(Number.isFinite(nextDuration) ? nextDuration : 0);
        }}
        onTimeUpdate={event => {
          setCurrentTime(event.currentTarget.currentTime);
          setIsPlaying(!event.currentTarget.paused);
        }}
        onPause={() => setIsPlaying(false)}
        onEnded={() => {
          setIsPlaying(false);
          setCurrentTime(duration);
        }}
      />
      <div className="flex min-w-0 items-center gap-2">
        <Music className="h-4 w-4 shrink-0 text-emerald-300" />
        <div className="min-w-0">
          <p className="text-[10px] font-semibold leading-tight text-emerald-200">
            {label}
          </p>
          {fileName && (
            <p className="truncate text-[9px] leading-tight text-emerald-100/70">
              {fileName}
            </p>
          )}
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2 rounded-full bg-black/35 px-2 py-1.5">
        <button
          type="button"
          onClick={togglePlayback}
          disabled={!src}
          className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-emerald-300 text-black transition-colors hover:bg-emerald-200 disabled:cursor-not-allowed disabled:opacity-40"
          title={isPlaying ? "Pause audio" : "Play audio"}
        >
          {isPlaying ? (
            <Pause className="h-3 w-3" />
          ) : (
            <Play className="h-3 w-3 translate-x-[0.5px] fill-current" />
          )}
        </button>
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={duration > 0 ? Math.min(currentTime, duration) : 0}
          onChange={handleScrub}
          disabled={!src || duration <= 0}
          className="h-1.5 min-w-0 flex-1 cursor-pointer appearance-none rounded-full bg-emerald-950 disabled:cursor-not-allowed disabled:opacity-40 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-200 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-emerald-200"
          style={{ background: rangeBackground }}
          aria-label="Audio preview position"
        />
        <span className="w-14 text-right text-[9px] tabular-nums text-emerald-100/70">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}
