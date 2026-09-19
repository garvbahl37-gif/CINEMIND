/** Trim to a length without slicing a word in half. */
export function clip(text: string, max: number): string {
  if (!text || text.length <= max) return text ?? '';
  const cut = text.slice(0, max);
  const space = cut.lastIndexOf(' ');
  return `${(space > max * 0.6 ? cut.slice(0, space) : cut).replace(/[\s,.;:—-]+$/, '')}…`;
}
