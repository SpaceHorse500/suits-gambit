# gui_play.py
# Pygame GUI that REUSES your existing classes:
# - game.SuitsGambitGame (with "forbid after every draw" logic)
# - players.BasePlayer
# - your control bot MetaOne (or HandPlayer fallback)
#
# Run:  python gui_play.py

import sys
import math
import random
import pygame

# ---- reuse your code ----
from game import SuitsGambitGame
from players import BasePlayer
from cards import SUITS  # your SUITS = ["♣","♦","♥","♠"]

# Try to reuse your control bot(s)
MetaBotClass = None
try:
    # if you already defined MetaOne somewhere (e.g., ga_controls or a bot module), import it
    from ga.ga_controls import MetaOne as MetaBotClass  # adjust if you keep MetaOne elsewhere
except Exception:
    try:
        # or fallback to the HandPlayer you showed earlier
        from smart_player import HandPlayer as MetaBotClass
    except Exception:
        MetaBotClass = None


# ---------- small UI helpers ----------
W, H = 1024, 680
FPS = 60
WHITE = (246, 246, 246)
BLACK = (24, 24, 24)
GREY = (190, 190, 190)
DGREY = (120, 120, 120)
BLUE = (40, 120, 220)
GREEN = (34, 150, 90)
RED = (200, 60, 60)
YELLOW = (230, 180, 40)
PURPLE = (130, 70, 200)

BTN_W, BTN_H = 140, 46

class Button:
    def __init__(self, rect, text, color, text_color=WHITE, font=None, tag=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = font
        self.tag = tag
        self.enabled = True

    def draw(self, surf):
        c = self.color if self.enabled else DGREY
        pygame.draw.rect(surf, c, self.rect, border_radius=10)
        if self.font:
            label = self.font.render(self.text, True, self.text_color)
            surf.blit(label, label.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


# ---------- Pygame UI mediator (blocking prompts) ----------
class PygameUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Suits Gambit — Human vs Bot")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.huge = pygame.font.SysFont("consolas", 46, bold=True)

    def _blit_text(self, txt, x, y, big=False):
        f = self.big if big else self.font
        img = f.render(txt, True, BLACK)
        self.screen.blit(img, (x, y))

    def _draw_frame(self, title):
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, GREY, (20, 20, W-40, 80), border_radius=12, width=2)
        self._blit_text(title, 40, 45, big=True)

    def _draw_scores(self, human, bot):
        y = 120
        pygame.draw.rect(self.screen, GREY, (20, y, W-40, 110), border_radius=12, width=2)
        self._blit_text(f"You scores: {human.round_scores}", 40, y+15)
        self._blit_text(f"You ops   : {human.ops_between}", 40, y+45)
        self._blit_text(f"Bot scores: {bot.round_scores}", 40, y+75)
        self._blit_text(f"Bot ops   : {bot.ops_between}", 40, y+105)

    def _draw_card_and_odds(self, center_text, ctx, current_forbidden=None):
        # info card (forbidden pick) or last card (continue/stop)
        pygame.draw.rect(self.screen, GREY, (20, 250, W-40, 160), border_radius=12, width=2)
        self._blit_text(center_text, 40, 270, big=True)

        rem_by = (ctx.get("deck_remaining_by_suit") or {})
        suits_line = "  |  ".join([f"{s}:{rem_by.get(s,0)}" for s in SUITS])
        self._blit_text("Deck remaining by suit: " + suits_line, 40, 310)

        if current_forbidden:
            total_remaining = sum(rem_by.values()) or 1
            forb_left = rem_by.get(current_forbidden, 0)
            p_bust = forb_left / total_remaining
            self._blit_text(f"Current forbidden: {current_forbidden}  |  p(bust next) ≈ {p_bust:.3f}", 40, 340)

    def ask_forbidden(self, round_idx, human, bot, info_card, ctx) -> str:
        # suit buttons
        buttons = []
        sx = 80
        for s in SUITS:
            b = Button((sx, 450, BTN_W, BTN_H), s, BLUE if s in ("♣","♠") else RED, WHITE, self.big, tag=("suit", s))
            buttons.append(b)
            sx += BTN_W + 20

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    pos = ev.pos
                    for b in buttons:
                        if b.hit(pos):
                            return b.tag[1]

            self._draw_frame(f"Round {round_idx}: Choose a forbidden suit for the NEXT draw")
            self._draw_scores(human, bot)
            self._draw_card_and_odds(f"Info card shown: {info_card['rank']}{info_card['suit']}", ctx)
            for b in buttons:
                b.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)

    def ask_continue(self, round_idx, human, bot, ctx) -> str:
        cont = Button((W//2 - BTN_W - 20, 450, BTN_W, BTN_H), "Continue", GREEN, WHITE, self.big, tag=("go", "continue"))
        stop = Button((W//2 + 20, 450, BTN_W, BTN_H), "Stop", RED, WHITE, self.big, tag=("go", "stop"))
        buttons = [cont, stop]
        last = ctx.get("last_card") or {}
        last_s = f"{last.get('rank','?')}{last.get('suit','?')}"
        forb = ctx.get("current_forbidden")

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    pos = ev.pos
                    for b in buttons:
                        if b.hit(pos):
                            return b.tag[1]

            self._draw_frame(f"Round {round_idx}: Continue or Stop?")
            self._draw_scores(human, bot)
            self._draw_card_and_odds(f"Last card: {last_s}   |   Points so far: {ctx.get('current_points',0)}",
                                     ctx, current_forbidden=forb)
            for b in buttons:
                b.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)

    def ask_operator(self, between_text, human, bot) -> str:
        plus = Button((W//2 - BTN_W - 20, 450, BTN_W, BTN_H), "+", BLUE, WHITE, self.big, tag=("op", "+"))
        times = Button((W//2 + 20, 450, BTN_W, BTN_H), "×", PURPLE, WHITE, self.big, tag=("op", "×"))
        buttons = [plus, times]

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    pos = ev.pos
                    for b in buttons:
                        if b.hit(pos):
                            return b.tag[1]

            self._draw_frame(between_text)
            self._draw_scores(human, bot)
            for b in buttons:
                b.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)

    def show_final(self, results, winner_name):
        # Simple final screen
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return

            self.screen.fill(WHITE)
            self._blit_text("=== FINAL ===", 40, 40, big=True)
            y = 100
            for name, val in results.items():
                self._blit_text(f"{name}: {val}", 60, y)
                y += 36
            self._blit_text(f"Winner: {winner_name if winner_name else 'Tie!'}  (press Enter to close)", 40, y+20, big=True)
            pygame.display.flip()
            self.clock.tick(FPS)


# ---------- Human adapter reusing your BasePlayer ----------
class HumanPlayer(BasePlayer):
    """
    A BasePlayer that blocks to get choices from the Pygame UI.
    Works with your SuitsGambitGame as-is.
    """
    def __init__(self, name: str, ui: PygameUI, me_ref=None, opp_ref=None):
        super().__init__(name)
        self.ui = ui
        # direct references so UI can show live scores while picking
        self._me_ref = me_ref if me_ref is not None else self
        self._opp_ref = opp_ref

    def choose_forbidden_suit(self, first_revealed, ctx):
        # ctx contains: info_card, deck_remaining_by_suit, etc.
        info = ctx.get("info_card") or {"rank": "?", "suit": "?"}
        r = int(ctx.get("round_index", 1))
        return self.ui.ask_forbidden(r, self._me_ref, self._opp_ref, info, ctx)

    def choose_continue_or_stop(self, current_points, ctx):
        r = int(ctx.get("round_index", 1))
        return self.ui.ask_continue(r, self._me_ref, self._opp_ref, ctx)

    def choose_operator_between_rounds(self, my_scores, all_scores, previous_picks, ctx):
        r = int(ctx.get("round_index", 1))
        return self.ui.ask_operator(f"Pick operator between R{r} and R{r+1}", self._me_ref, self._opp_ref)


# ---------- Fallback Meta bot (only if none found) ----------
class SafeMetaFallback(BasePlayer):
    """A conservative bot if MetaOne/HandPlayer import fails."""
    def __init__(self, name="MetaFallback"):
        super().__init__(name)
        self._forbidden = None

    def choose_forbidden_suit(self, first_revealed, ctx):
        rem = ctx.get("deck_remaining_by_suit") or {}
        if rem and all(s in rem for s in SUITS):
            m = min(rem[s] for s in SUITS)
            cands = [s for s in SUITS if rem[s] == m]
            self._forbidden = random.choice(cands)
        else:
            self._forbidden = first_revealed.suit
        return self._forbidden

    def choose_continue_or_stop(self, current_points, ctx):
        last_op = self.ops_between[-1] if self.ops_between else "+"
        rem_by = ctx.get("deck_remaining_by_suit") or {}
        total = sum(rem_by.values()) or 1
        forb = ctx.get("current_forbidden") or self._forbidden
        p_bust = (rem_by.get(forb, 0) / total) if forb else 0.25

        if last_op == "×":
            if current_points < 3 and p_bust <= 0.35:
                return "continue"
            thresh = 0.75 / (current_points + 1)
            return "stop" if p_bust >= thresh else "continue"
        else:
            if current_points < 3:
                return "continue"
            thresh = 1.0 / (current_points + 1)
            if p_bust <= 0.18:
                return "continue"
            return "stop" if p_bust >= thresh else "continue"

    def choose_operator_between_rounds(self, my_scores, all_scores, previous_picks, ctx):
        last = my_scores[-1] if my_scores else 0
        if last == 0:
            return "+"
        return "×" if (last >= 5 and random.random() < 0.6) else "+"


def main():
    ui = PygameUI()

    # Build players
    human = HumanPlayer("You", ui)
    if MetaBotClass is None:
        bot = SafeMetaFallback("MetaOne")
    else:
        bot = MetaBotClass("MetaOne")

    # Let the UI show live scores while picking
    human._opp_ref = bot

    # Run one game using your SuitsGambitGame (reused!)
    game = SuitsGambitGame([human, bot], verbose=0, seed=random.randrange(1_000_000))
    winner, results = game.play()

    ui.show_final(results, winner)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        print("GUI crashed:", e)
        pygame.quit()
        raise
