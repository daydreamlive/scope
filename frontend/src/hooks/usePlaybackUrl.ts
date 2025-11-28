import { create } from "zustand";

interface PlaybackUrlState {
  loading: boolean;
  playbackUrl: string | null;
  setPlaybackUrl: (url: string | null) => void;
  setLoading: (loading: boolean) => void;
}

export const usePlaybackUrl = create<PlaybackUrlState>(set => ({
  loading: true,
  playbackUrl: null,
  setPlaybackUrl: url => set({ playbackUrl: url }),
  setLoading: (loading: boolean) => set({ loading }),
}));
