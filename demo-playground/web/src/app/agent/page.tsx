'use client';

import dynamic from 'next/dynamic';
import Nav from '@/components/ui/Nav';
import HostedBanner from '@/components/ui/HostedBanner';
import { isStaticHosting } from '@/lib/hosting';

const AgentChat = dynamic(() => import('@/components/agent/AgentChat'), { ssr: false });

export default function AgentPage() {
  return (
    <>
      <Nav />
      <HostedBanner variant="agent" />
      {!isStaticHosting && <AgentChat />}
    </>
  );
}
