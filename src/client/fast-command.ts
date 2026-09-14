/** Build the locale resolver used by the /fast command contribution. */
export function fastCommandDescription(t: () => string): () => string {
  return () => t()
}
