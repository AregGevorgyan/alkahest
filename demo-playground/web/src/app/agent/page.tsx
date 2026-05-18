'use client';

import dynamic from 'next/dynamic';
import Nav from '@/components/ui/Nav';

const AgentChat = dynamic(() => import('@/components/agent/AgentChat'), { ssr: false });

export default function AgentPage() {
  return (
    <>
      <Nav />
      <AgentChat />
    </>
  );
}
