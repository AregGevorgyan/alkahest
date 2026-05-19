'use client';

import { SettingsProvider } from '@/components/ui/SettingsContext';

export function Providers({ children }: { children: React.ReactNode }) {
  return <SettingsProvider>{children}</SettingsProvider>;
}
