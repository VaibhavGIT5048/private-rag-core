import { ApiError } from '@/services/api'

/** The backend's `detail` string for an ApiError, or `fallback` for anything else
 * (network failure, abort, non-API throw). */
export function formatApiError(err: unknown, fallback: string): string {
  return err instanceof ApiError ? err.detail : fallback
}
