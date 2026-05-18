declare module 'katex/contrib/auto-render' {
  function renderMathInElement(
    element: HTMLElement,
    options?: {
      delimiters?: Array<{ left: string; right: string; display: boolean }>;
      throwOnError?: boolean;
    },
  ): void;
  export default renderMathInElement;
}
